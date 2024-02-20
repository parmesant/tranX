from transformers import T5Tokenizer, T5Config, T5Model, MistralModel, T5ForConditionalGeneration, T5EncoderModel
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from model.parser import Parser
from model.wikisql.dataset import WikiSqlBatch
from model.pointer_net import PointerNet
from itertools import chain
from model import nn_utils
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from asdl.hypothesis import Hypothesis
from asdl.lang.sql.sql_transition_system import WikiSqlSelectColumnAction
from asdl.transition_system import ApplyRuleAction, ReduceAction, GenTokenAction
from components.action_info import ActionInfo
from components.decode_hypothesis import DecodeHypothesis
import warnings
warnings.filterwarnings('ignore')

device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# torch.set_default_device(device)

def getLabels(example, prod_rules:dict):
    breakpoint()
    label = []
    tgt_actions = example.tgt_actions
    for action in tgt_actions:
        if 'Reduce' in action.__repr__():
            action_str = 'Reduce'
        elif 'GenToken' in action.__repr__():
            action_str = 'GenToken'
        elif 'SelectColumnAction' in action.__repr__():
            action_str = 'SelectColumnAction'
        else:
            # print(action)
            action_str = str(action.action.production)
        label.append(prod_rules[action_str])
    return label



def tokenize(example, prod_rules:dict):
    src_sent = example.src_sent
    table_header = example.table.header
    table_header_str = ""
    for column in table_header:
        name = column.name
        type = column.type
        table_header_str += name + " " + type + " ,"
    # remove the last comma
    table_header_str = table_header_str[:-1]
    prompt = "translate English to SQL: " + ' '.join(src_sent) + " table header: " + table_header_str
    # prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids

    # # get labels
    # label = getLabels(example, prod_rules)
    return prompt


class CustomT5(Parser):
    def __init__(self, args, vocab, transition_system, head_size) -> None:
        super(CustomT5,self).__init__(args, vocab, transition_system)
        self.args = args
        self.transition_system = transition_system
        self.vocab = vocab
        self.grammar = self.transition_system.grammar

        self.table_header_lstm = nn.LSTM(args.embed_size, int(args.hidden_size / 2), bidirectional=True, batch_first=True).to(device)

        self.column_pointer_net = PointerNet(args.hidden_size, args.hidden_size, attention_type=args.column_att).to(device)

        self.column_rnn_input = nn.Linear(args.hidden_size, args.embed_size, bias=False).to(device)

        self.hidden_size = 256
        self.max_sent_len = 100

        # self.uemat1 = torch.randn((768,self.hidden_size),requires_grad=True, device=device)
        # self.uemat2 = torch.randn((1,self.max_sent_len+2),requires_grad=True, device=device)

        self.lin = nn.Linear(768, self.hidden_size*2).to(device) # for T5

        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        # self.t5 = T5ForConditionalGeneration.from_pretrained("t5-base")
        # self.t5 = T5Model.from_pretrained("t5-base")
        self.encoder = T5EncoderModel.from_pretrained("t5-base").to(device)

        self.dropout = nn.Dropout(0.1).to(device)
        self.head = nn.Linear(768, head_size).to(device)

        # in case of multiple output labels, use sigmoid

    def encode_table_header(self, tables):
        # input, ids of table word: (batch_size, max_column_num)
        # encode_output: (max_head_word_num, batch_size, max_column_num, hidden_size)

        # (batch_size, max_column_num, max_head_word_num)
        # table_head_mask: (batch_size, max_column_num)
        # table_col_lens: (batch_size, max_column_num)
        table_head_wids, table_col_lens = WikiSqlBatch.get_table_header_input_tensor(tables,
                                                                                     self.vocab.source,
                                                                                     cuda=self.args.cuda)

        # hack: pack_padded_sequence requires seq length to be greater than 1
        for tbl in table_col_lens:
            for i in range(len(tbl)):
                if tbl[i] == 0: tbl[i] = 1

        table_header_mask = WikiSqlBatch.get_table_header_mask(tables, cuda=self.args.cuda)

        table_header_mask.to(device)
        table_head_wids.to(device)
        # (batch_size, max_column_num, max_head_word_num, word_embed_size)
        table_head_word_embeds = self.src_embed(table_head_wids.view(-1)).view(list(table_head_wids.size()) + [self.src_embed.embedding_dim])

        batch_size = table_head_word_embeds.size(0)
        max_col_num = table_head_word_embeds.size(1)
        max_col_word_num = table_head_word_embeds.size(2)

        # (batch_size * max_column_num, max_head_word_num, word_embed_size)
        table_head_word_embeds_flatten = table_head_word_embeds.view(batch_size * max_col_num,
                                                                     max_col_word_num, -1)
        table_col_lens_flatten = list(chain.from_iterable(table_col_lens))
        sorted_col_ids = sorted(list(range(len(table_col_lens_flatten))), key=lambda x: -table_col_lens_flatten[x])
        sorted_table_col_lens_flatten = [table_col_lens_flatten[i] for i in sorted_col_ids]

        col_old_pos_map = [-1] * len(sorted_col_ids)
        for new_pos, old_pos in enumerate(sorted_col_ids):
            col_old_pos_map[old_pos] = new_pos

        # (batch_size * max_column_num, max_head_word_num, word_embed_size)
        sorted_table_head_word_embeds = table_head_word_embeds_flatten[sorted_col_ids, :, :]

        packed_table_head_word_embeds = pack_padded_sequence(sorted_table_head_word_embeds, sorted_table_col_lens_flatten, batch_first=True)

        # column_word_encodings: (batch_size * max_column_num, max_head_word_num, hidden_size)
        column_word_encodings, (table_header_encoding, table_head_last_cell) = self.table_header_lstm(packed_table_head_word_embeds)
        column_word_encodings, _ = pad_packed_sequence(column_word_encodings, batch_first=True)

        # (batch_size * max_column_num, max_head_word_num, hidden_size)
        column_word_encodings = column_word_encodings[col_old_pos_map]
        # (batch_size, max_column_num, max_head_word_num, hidden_size)
        column_word_encodings = column_word_encodings.view(batch_size, max_col_num, max_col_word_num, -1)

        # (batch_size, hidden_size * 2)
        table_header_encoding = torch.cat([table_header_encoding[0], table_header_encoding[1]], -1)
        # table_head_last_cell = torch.cat([table_head_last_cell[0], table_head_last_cell[1]], -1)

        # same
        table_header_encoding = table_header_encoding[col_old_pos_map]
        # (batch_size, max_column_num, hidden_size)
        table_header_encoding = table_header_encoding.view(batch_size, max_col_num, -1)

        return column_word_encodings, table_header_encoding, table_header_mask

    def forward(self, input_ids=None, attention_mask=None, labels=None) -> None:
        
        # # Add custom layers
        # sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state
        
        # # pass the outputs through the head
        # logits = self.head(sequence_output[:,0,:].view(-1,768)) # calculate losses
        
        # loss = None
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=logits,
        #     # past_key_values=outputs.past_key_values,
        #     # decoder_hidden_states=outputs.decoder_hidden_states,
        #     # decoder_attentions=outputs.decoder_attentions,
        #     # cross_attentions=outputs.cross_attentions,
        #     # encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        #     # encoder_hidden_states=outputs.encoder_hidden_states,
        #     # encoder_attentions=outputs.encoder_attentions,
        # )

        # PointerNet accepts 3 params during training and 2 during scoring
        # training-
        # scoring-
        # utterance_encoding (batch_size, src_sent_len, hidden_size)
        # att_t (batch_size, hidden_size)
        # so we decide an arbitrary src_sent_len and fill the sentences with <pad>
        pass

    
    def getUtteranceEncoding(self, input_ids=None):
        # we want to convert
        # (num_tokens,768) -> (max_sent_len+2, hidden_size) (sentences + last_state + last_cell)
        
        # get outputs from T5
        outputs = self.encoder(input_ids=input_ids,output_hidden_states=True)
        # print(f"{len(outputs.hidden_states)=}") # 13 layers
        # for h in outputs.hidden_states:
        #     print(f"{h.shape=}")
        outputs = outputs.last_hidden_state
        # # shape is (batch_size, max_num_tokens, hidden_size) batch_size is 1 here hence squeeze
        # outputs = torch.squeeze(outputs)
        # # shape is (num_tokens, hidden_size)

        # lose information by avg?
        # convert output to shape (batch_size, hidden_size)
        outputs = outputs.mean(dim=1, keepdim=True)

        outputs = self.lin(outputs)
        outputs = torch.squeeze(outputs,1)
        print(f"lin {outputs.shape=}")
        print(f"{input_ids=}")
        # outputs = torch.matmul(outputs,self.uemat1)
        # # outputs = (1, hidden_size)

        # outputs = torch.matmul(outputs.T,self.uemat2).T
        # # # outputs = (max_sent_len+1, hidden_size)
        return outputs
        
    
    def score(self, examples, prod_rules, tokenizer, device):
        batch_examples_modified = [tokenize(e,prod_rules) for e in examples]
        batch_examples_modified = tokenizer(batch_examples_modified,return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        batch_examples_modified = batch_examples_modified.to(device)

        labels = [getLabels(e,prod_rules) for e in examples]

    def parse(self, example, context, beam_size=5):
        # breakpoint()
        production_len = 12
        primitive_len = 5
        field_len = 5
        type_len = 6
        src_len = 14838
        question = ' '.join(example.src_sent)
        print(f"{question=}")
        table = context
        args = self.args
        input_sentence = question
        input_ids = self.tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)

        src_sent_var = nn_utils.to_input_variable([example.src_sent], self.vocab.source,
                                                  cuda=self.args.cuda, training=False)
        
        utterance_encodings, (last_state, last_cell) = self.encode(src_sent_var, [len(example.src_sent)])

        encodings = self.getUtteranceEncoding(input_ids) # convert this into last_state and last_cell
        last_state = encodings[:,0:self.hidden_size]
        last_cell = encodings[:,self.hidden_size:]
        # print(f"{encodings.shape=}\n{utterance_encodings.shape=}\n{last_cell.shape=}\n{last_state.shape=}\n{question_size=}")

        dec_init_vec = self.init_decoder_state(last_state, last_cell)

        column_word_encodings, table_header_encoding, table_header_mask = self.encode_table_header([table])

        past_column_encodings = dict()

        h_tm1 = dec_init_vec
        # (batch_size, query_len, hidden_size)
        utterance_encodings_att_linear = self.att_src_linear(utterance_encodings)

        zero_action_embed = Variable(self.new_tensor(self.args.action_embed_size).zero_())

        t = 0
        hypotheses = [DecodeHypothesis()]
        hyp_states = [[]]
        completed_hypotheses = []
        with torch.no_grad():
            while len(completed_hypotheses) < beam_size and t < self.args.decode_max_time_step:
                hyp_num = len(hypotheses)
                # print(f"{hyp_num=},{question=},{t=}")
                # (hyp_num, src_sent_len, hidden_size * 2)
                exp_src_encodings = utterance_encodings.expand(hyp_num, utterance_encodings.size(1), utterance_encodings.size(2))
                # (hyp_num, src_sent_len, hidden_size)
                exp_src_encodings_att_linear = utterance_encodings_att_linear.expand(hyp_num,
                                                                                    utterance_encodings_att_linear.size(1),
                                                                                    utterance_encodings_att_linear.size(2))

                # x: [prev_action, parent_production_embed, parent_field_embed, parent_field_type_embed, parent_action_state]
                if t == 0:
                    x = Variable(self.new_tensor(1, self.decoder_lstm.input_size).zero_(), volatile=True)

                    if args.no_parent_field_type_embed is False:
                        offset = args.action_embed_size  # prev_action
                        offset += args.hidden_size * (not args.no_input_feed)
                        offset += args.action_embed_size * (not args.no_parent_production_embed)
                        offset += args.field_embed_size * (not args.no_parent_field_embed)

                        x[0, offset: offset + args.type_embed_size] = \
                            self.type_embed.weight[self.grammar.type2id[self.grammar.root_type]]
                else:
                    a_tm1_embeds = []
                    for e_id, hyp in enumerate(hypotheses):
                        action_tm1 = hyp.actions[-1]
                        if action_tm1:
                            if isinstance(action_tm1, ApplyRuleAction):
                                a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                            elif isinstance(action_tm1, ReduceAction):
                                a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                            elif isinstance(action_tm1, WikiSqlSelectColumnAction):
                                a_tm1_embed = self.column_rnn_input(table_header_encoding[0, action_tm1.column_id])
                            elif isinstance(action_tm1, GenTokenAction):
                                a_tm1_embed = self.src_embed.weight[self.vocab.source[action_tm1.token]]
                            else:
                                raise ValueError('unknown action %s' % action_tm1)
                        else:
                            a_tm1_embed = zero_action_embed

                        a_tm1_embeds.append(a_tm1_embed)

                    a_tm1_embeds = torch.stack(a_tm1_embeds)

                    inputs = [a_tm1_embeds]
                    if args.no_input_feed is False:
                        inputs.append(att_tm1)
                    if args.no_parent_production_embed is False:
                        # frontier production
                        frontier_prods = [hyp.frontier_node.production for hyp in hypotheses]
                        frontier_prod_embeds = self.production_embed(Variable(self.new_long_tensor(
                            [self.grammar.prod2id[prod] for prod in frontier_prods])))
                        inputs.append(frontier_prod_embeds)
                    if args.no_parent_field_embed is False:
                        # frontier field
                        frontier_fields = [hyp.frontier_field.field for hyp in hypotheses]
                        frontier_field_embeds = self.field_embed(Variable(self.new_long_tensor([
                            self.grammar.field2id[field] for field in frontier_fields])))

                        inputs.append(frontier_field_embeds)
                    if args.no_parent_field_type_embed is False:
                        # frontier field type
                        frontier_field_types = [hyp.frontier_field.type for hyp in hypotheses]
                        frontier_field_type_embeds = self.type_embed(Variable(self.new_long_tensor([
                            self.grammar.type2id[type] for type in frontier_field_types])))
                        inputs.append(frontier_field_type_embeds)

                    # parent states
                    if args.no_parent_state is False:
                        p_ts = [hyp.frontier_node.created_time for hyp in hypotheses]
                        parent_states = torch.stack([hyp_states[hyp_id][p_t][0] for hyp_id, p_t in enumerate(p_ts)])
                        parent_cells = torch.stack([hyp_states[hyp_id][p_t][1] for hyp_id, p_t in enumerate(p_ts)])

                        if args.lstm == 'parent_feed':
                            h_tm1 = (h_tm1[0], h_tm1[1], parent_states, parent_cells)
                        else:
                            inputs.append(parent_states)

                    x = torch.cat(inputs, dim=-1)

                (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_encodings,
                                                exp_src_encodings_att_linear,
                                                src_token_mask=None)
                # print(f"{utterance_encodings.shape=}\n{att_t.shape=}\n{last_state.shape=}\n{last_cell.shape=}")
                
                # ApplyRule action probability
                # (batch_size, grammar_size)
                apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

                # column attention
                # (batch_size, max_head_num)
                column_attention_weights = self.column_pointer_net(table_header_encoding, table_header_mask,att_t.unsqueeze(0)).squeeze(0)

                column_selection_log_prob = torch.log(column_attention_weights)
                # print(f"{len(hypotheses)=} {t=} {column_selection_log_prob.shape=} {att_t.shape=}")
                # print(column_selection_log_prob)

                # (batch_size, 2)
                primitive_predictor_prob = F.softmax(self.primitive_predictor(att_t), dim=-1)

                # primitive copy prob
                # (batch_size, src_token_num)
                primitive_copy_prob = self.src_pointer_net(utterance_encodings, None,
                                                        att_t.unsqueeze(0)).squeeze(0)

                # (batch_size, primitive_vocab_size)
                primitive_gen_from_vocab_prob = F.softmax(self.tgt_token_readout(att_t), dim=-1)

                new_hyp_meta = []

                # hypothesis is like a transition system for parsing
                for hyp_id, hyp in enumerate(hypotheses):
                    # generate new continuations
                    action_types = self.transition_system.get_valid_continuation_types(hyp)
                    # print(f"{action_types=},{hyp_id=}")
                    # in the first go, we get ApplyRuleAction

                    for action_type in action_types:
                        if action_type == ApplyRuleAction:
                            productions = self.transition_system.get_valid_continuating_productions(hyp)
                            for production in productions:
                                prod_id = self.grammar.prod2id[production]
                                prod_score = apply_rule_log_prob[hyp_id, prod_id]
                                new_hyp_score = hyp.score + prod_score

                                meta_entry = {'action_type': 'apply_rule', 'prod_id': prod_id,
                                            'score': prod_score, 'new_hyp_score': new_hyp_score,
                                            'prev_hyp_id': hyp_id}
                                new_hyp_meta.append(meta_entry)
                        elif action_type == ReduceAction:
                            action_score = apply_rule_log_prob[hyp_id, len(self.grammar)]
                            new_hyp_score = hyp.score + action_score

                            meta_entry = {'action_type': 'apply_rule', 'prod_id': len(self.grammar),
                                        'score': action_score, 'new_hyp_score': new_hyp_score,
                                        'prev_hyp_id': hyp_id}
                            new_hyp_meta.append(meta_entry)
                        elif action_type == WikiSqlSelectColumnAction:
                            for col_id, column in enumerate(table.header):
                                # print(f"{t=}, {hyp_id=}, {column_selection_log_prob.shape=}")
                                col_sel_score = column_selection_log_prob[hyp_id, col_id]
                                new_hyp_score = hyp.score + col_sel_score

                                meta_entry = {'action_type': 'sel_col', 'col_id': col_id,
                                            'score': col_sel_score, 'new_hyp_score': new_hyp_score,
                                            'prev_hyp_id': hyp_id}
                                new_hyp_meta.append(meta_entry)
                        elif action_type == GenTokenAction:
                            print('entered gentoken')
                            # remember that we can only copy stuff from the input!
                            # we only copy tokens sequentially!!
                            prev_action = hyp.action_infos[-1].action

                            valid_token_pos_list = []
                            if type(prev_action) is GenTokenAction and \
                                    not prev_action.is_stop_signal():
                                token_pos = hyp.action_infos[-1].src_token_position + 1
                                if token_pos < len(question):
                                    valid_token_pos_list = [token_pos]
                            else:
                                valid_token_pos_list = list(range(len(question)))

                            col_id = hyp.frontier_node['col_idx'].value
                            if table.header[col_id].type == 'real':
                                valid_token_pos_list = [i for i in valid_token_pos_list
                                                        if any(c.isdigit() for c in question[i]) or
                                                        hyp._value_buffer and question[i] in (',', '.', '-', '%')]

                            p_copies = primitive_predictor_prob[hyp_id, 1] * primitive_copy_prob[hyp_id]
                            print(f"{valid_token_pos_list=}\n{p_copies.shape=}\n\n")
                            for token_pos in valid_token_pos_list:
                                token = question[token_pos]
                                p_copy = p_copies[token_pos]
                                score_copy = torch.log(p_copy)

                                meta_entry = {'action_type': 'gen_token',
                                            'token': token, 'token_pos': token_pos,
                                            'score': score_copy, 'new_hyp_score': score_copy + hyp.score,
                                            'prev_hyp_id': hyp_id}
                                new_hyp_meta.append(meta_entry)

                            # add generation probability for </primitive>
                            if hyp._value_buffer:
                                eos_prob = primitive_predictor_prob[hyp_id, 0] * \
                                        primitive_gen_from_vocab_prob[hyp_id, self.vocab.primitive['</primitive>']]
                                eos_score = torch.log(eos_prob)

                                meta_entry = {'action_type': 'gen_token',
                                            'token': '</primitive>',
                                            'score': eos_score, 'new_hyp_score': eos_score + hyp.score,
                                            'prev_hyp_id': hyp_id}
                                new_hyp_meta.append(meta_entry)

                if not new_hyp_meta: break

                new_hyp_scores = torch.cat([x['new_hyp_score'].resize(1) for x in new_hyp_meta])
                top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                        k=min(new_hyp_scores.size(0),
                                                                beam_size - len(completed_hypotheses)))

                live_hyp_ids = []
                new_hypotheses = []
                # breakpoint()
                for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                    action_info = ActionInfo()
                    hyp_meta_entry = new_hyp_meta[meta_id]
                    prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                    prev_hyp = hypotheses[prev_hyp_id]

                    action_type_str = hyp_meta_entry['action_type']
                    # print(f"{action_type_str=}, {len(top_new_hyp_scores)=}")
                    if action_type_str == 'apply_rule':
                        # ApplyRule action
                        prod_id = hyp_meta_entry['prod_id']
                        if prod_id < len(self.grammar):
                            production = self.grammar.id2prod[prod_id]
                            action = ApplyRuleAction(production)
                        # Reduce action
                        else:
                            action = ReduceAction()
                    elif action_type_str == 'sel_col':
                        action = WikiSqlSelectColumnAction(hyp_meta_entry['col_id'])
                    else:
                        action = GenTokenAction(hyp_meta_entry['token'])
                        if 'token_pos' in hyp_meta_entry:
                            action_info.copy_from_src = True
                            action_info.src_token_position = hyp_meta_entry['token_pos']

                    action_info.action = action
                    action_info.t = t

                    if t > 0:
                        action_info.parent_t = prev_hyp.frontier_node.created_time
                        action_info.frontier_prod = prev_hyp.frontier_node.production
                        action_info.frontier_field = prev_hyp.frontier_field.field

                    new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                    new_hyp.score = new_hyp_score

                    if new_hyp.completed:
                        # print('t=',t)
                        completed_hypotheses.append(new_hyp)
                    else:
                        new_hypotheses.append(new_hyp)
                        live_hyp_ids.append(prev_hyp_id)
                        # for i,nh in enumerate(new_hypotheses):
                        #     print(f"{i} {nh.actions=}")
                        # print()
                        # # print(f"{question=},{len(new_hypotheses)=}")

                if live_hyp_ids:
                    hyp_states = [hyp_states[i] + [(h_t[i], cell_t[i])] for i in live_hyp_ids]
                    h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                    att_tm1 = att_t[live_hyp_ids]
                    hypotheses = new_hypotheses
                    t += 1
                else: break

        completed_hypotheses.sort(key=lambda hyp: -hyp.score)
        # breakpoint()
        return completed_hypotheses