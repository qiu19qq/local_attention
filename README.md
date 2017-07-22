A TensorFlow Implementation of the “Local Attention” in paper: Effective Approaches to Attention-based Neural Machine Translation    
====

Why this project?    
－－－－－－
  Attention is a useful machenism in NMT. Recently, the paper "Attention ia all you need" shows a new network 
  architecture which based solely on attention machenisms. I have two goals with this project. One is to have a 
  full understanding of attention machenism and the paper "Effective Approaches to Attention-based Neural Machine Translation".
  Another is to implement "local attention" by using tensorflow, since I didn't find the corresponding function in tensorflow.
  
File description   
－－－
  data_utils.py   data preparation, the same one with Tensorflow: tutorials/rnn/translate/data_tuils.py  －
  
  seq2seq_model.py  the same one with Tensorflow: tutorials/rnn/translate/data_tuils.py  
  
  translate.py  the same one with Tensorflow: tutorials/rnn/translate/data_tuils.py  
  
  seq2seq_local.py  attention decoeder function. Implement the "local attention" in function "local_attention_decoder". Another attention function, implements to calculate the new alignment weights with Gaussian distribution but not sets
the window, is named "local_attention_decodere_nowindow". Function "attention_decoder" is the original one
in tensorflow.
                   
Training  
－－－
  Just use the same command line in Tensorflow tutorial:  
  python translate.py  
  
  --data_dir [your_data_directory] --train_dir [checkpoints_directory]  
  
  --en_vocab_size=40000 --fr_vocab_size=40000  
  
  

                    
