import tensorflow as tf
import run_classifier
import json
import tokenization

def read_classifier_examples(input_file, labels, is_training):
  """Read a SQuAD json file into a list of SquadExample."""
  with tf.gfile.Open(input_file, "r") as reader:
    input_data = json.load(reader)["data"]
  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
       return True
    return False
  examples = []
  for entry in input_data:
     for paragraph in entry["paragraphs"]:
        paragraph_text = paragraph["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        for qa in paragraph["qas"]:
            qas_id = qa["id"]
            question_text = qa["question"]
            start_position = None
            end_position = None
            orig_answer_text = None
            is_impossible =  False
            #if (len(qa["answers"]) != 1):
                
            if not is_impossible:
                answer = qa["answers"][0]
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length -
                                                   1]
                # Only add answers where the text can be exactly recovered from the
                # document. If this CAN'T happen it's likely due to weird Unicode
                # stuff so we will just skip the example.
                #
                # Note that this means for training mode, every example is NOT
                # guaranteed to be preserved.
                actual_text = " ".join(
                    doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = " ".join(
                    tokenization.whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    #tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                    #actual_text, cleaned_answer_text)
                    continue
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ""
            label = labels[qas_id]
            example = run_classifier.InputExample(guid=qas_id, text_a=question_text, text_b=orig_answer_text, label=label)
            examples.append(example)
  return examples
