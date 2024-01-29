from transformers import T5Tokenizer, T5Config, T5Model, MistralModel, T5ForConditionalGeneration, T5EncoderModel
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from asdl.hypothesis import Hypothesis
from asdl.lang.sql.sql_transition_system import WikiSqlSelectColumnAction
from asdl.transition_system import ApplyRuleAction, ReduceAction, GenTokenAction
from components.action_info import ActionInfo
from components.decode_hypothesis import DecodeHypothesis

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


class CustomT5(nn.Module):
    def __init__(self, args, vocab, transition_system, head_size) -> None:
        super(CustomT5,self).__init__()
        self.args = args
        self.transition_system = transition_system
        self.vocab = vocab
        self.grammar = self.transition_system.grammar

        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        # self.t5 = T5ForConditionalGeneration.from_pretrained("t5-base")
        # self.t5 = T5Model.from_pretrained("t5-base")
        self.encoder = T5EncoderModel.from_pretrained("t5-base")
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(768, head_size)

        # in case of multiple output labels, use sigmoid

    def forward(self, prod_rules, input_ids=None, attention_mask=None, labels=None) -> None:
        
        # get outputs from T5
        outputs = self.encoder(input_ids=input_ids)
        
        #Add custom layers
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state
        
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
    
    def score(self, examples, prod_rules, tokenizer, device):
        batch_examples_modified = [tokenize(e,prod_rules) for e in examples]
        batch_examples_modified = tokenizer(batch_examples_modified,return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        batch_examples_modified = batch_examples_modified.to(device)

        labels = [getLabels(e,prod_rules) for e in examples]

    def parse(self, example, table, beam_size):
        breakpoint()
        production_len = 12
        primitive_len = 5
        field_len = 5
        type_len = 6
        src_len = 14838
        question = ' '.join(example.src_sent)
        table = example.table
        input_sentence = "translate English to SQL: " + question + " table header: " + table
        input_ids = self.tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        utterance_encodings = self.encoder(input_ids=input_ids) # shape = (batch ,sentence length, 768)

        t = 0
        hypotheses = [DecodeHypothesis()]
        hyp_states = [[]]
        completed_hypotheses = []

        with torch.no_grad():
            while len(completed_hypotheses) < beam_size and t < self.args.decode_max_time_step:
                hyp_num = len(hypotheses)

                if t==0:
                    pass
                else:
                    pass


                # pass utterance encoding to decoder
                apply_rule_log_prob = F.log_softmax(self.head(utterance_encodings[0][:,:,:].view(-1,768)), dim=-1)[:,:production_len].mean(dim=0, keepdim=True)
                
                primitive_predictor_prob = F.softmax(self.head(utterance_encodings[0][:,:,:].view(-1,768)), dim=-1)[:,production_len:production_len+primitive_len].mean(dim=0, keepdim=True)

                column_selection_log_prob = F.log_softmax(self.head(utterance_encodings[0][:,:,:].view(-1,768)), dim=-1)[:,-src_len:].mean(dim=0, keepdim=True)

                new_hyp_meta = []
                
                for hyp_id, hyp in enumerate(hypotheses):
                    action_types = self.transition_system.get_valid_continuation_types(hyp)

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

                        # what to do about this???
                        elif action_type == WikiSqlSelectColumnAction:
                            for col_id, column in enumerate(table.header):
                                col_sel_score = column_selection_log_prob[hyp_id, col_id]
                                new_hyp_score = hyp.score + col_sel_score

                                meta_entry = {'action_type': 'sel_col', 'col_id': col_id,
                                            'score': col_sel_score, 'new_hyp_score': new_hyp_score,
                                            'prev_hyp_id': hyp_id}
                                new_hyp_meta.append(meta_entry)

                        # what to do about this??
                        elif action_type == GenTokenAction:
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

                            p_copies = primitive_predictor_prob[hyp_id, 1]# * primitive_copy_prob[hyp_id]
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
                for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                    action_info = ActionInfo()
                    hyp_meta_entry = new_hyp_meta[meta_id]
                    prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                    prev_hyp = hypotheses[prev_hyp_id]

                    action_type_str = hyp_meta_entry['action_type']
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
                        completed_hypotheses.append(new_hyp)
                    else:
                        new_hypotheses.append(new_hyp)
                        live_hyp_ids.append(prev_hyp_id)

                # what to do about this???
                if live_hyp_ids:
                    # hyp_states = [hyp_states[i] + [(h_t[i], cell_t[i])] for i in live_hyp_ids]
                    # h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                    # att_tm1 = att_t[live_hyp_ids]
                    # hypotheses = new_hypotheses
                    t += 1
                else: break
            
        completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses