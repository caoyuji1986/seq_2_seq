## Experiment Result
|  ckpt | lr_strategy |batch_size | opt |config|BLUE|
|:---   |:--------|:------- |:---- |:---|:---|
| -     |1       | bucket:10 tokens:6500 | Adam |1|BLEU = 7.75, 36.2/12.2/5.0/2.3|
| -     |1       | bucket:10 tokens:6500 | Adam |1 {"use_residual": true}|BLEU = 7.97, 36.2/12.3/5.1/2.3|
| -     |1       | bucket:10 tokens:6500 | Adam |1 {"use_residual": true，"num_hidden_layers":3}|BLEU = 8.67, 37.6/12.9/5.5/2.7|
| -     |1       | bucket:10 tokens:6500 | Adam |1 {"use_residual": true，"num_hidden_layers":3}, share decode embedding with project|BLEU = 9.37, 38.7/13.9/5.9/2.9|
| 4     |1       | bucket:10 tokens:6500 | Adam |1 {"use_residual": true，"num_hidden_layers":4}, share decode embedding with project|BLEU = 9.90, 40.0/14.8/6.6/3.3|
| 5     |1       | bucket:10 tokens:6500 | Adam |1 {"use_residual": true，"num_hidden_layers":4, "embedding_dropout_prob":0.2}, share decode embedding with project|BLEU = 11.71, 41.5/16.7/7.8/4.0|
| 6     |1       | bucket:10 tokens:6500  | AdamW |1 {"use_residual": true，"num_hidden_layers":4, "embedding_dropout_prob":0.2}, share decode embedding with project|BLEU = 12.98, 41.9/17.6/8.5/4.5 |

## LR STRATEGY

    ```1. python
    def create_train_opt_with_clip(loss, step_num_in_epoch=1000):
       global_steps_ = tf.train.get_or_create_global_step()
       global_step = tf.cast(x=global_steps_, dtype=tf.float32)
       #learning_rate = noam_scheme_seq_2_seq(global_step=global_step, step_num_in_epoch=step_num_in_epoch)
       learning_rate = noam_scheme(init_lr=0.003, global_step=global_step)
       # 论文中使用的就是这个优化器
       optimizer = tf.train.AdamOptimizer(learning_rate)
       grads, variables = zip(*optimizer.compute_gradients(loss))
       grads, global_norm = tf.clip_by_global_norm(grads, 5.0)
       train_op = optimizer.apply_gradients(grads_and_vars=zip(grads, variables), global_step=global_steps_)
       tf.summary.scalar('learning_rate', learning_rate)
       summaries = tf.summary.merge_all()
       return train_op, learning_rate
    ```
## Config
    
    ```1. json
    {   "vocab_size": 4000,
        "hidden_size": 1000,
        "cell_dropout_prob": 0.2,
        "attention_dropout_prob": 0.0,
        "embedding_dropout_prob": 0.0,
        "cell_type": "lstm",
        "use_attention": false,
        "num_hidden_layers":2,
        "use_residual": false,
        "reverse_encode_input": true
    }
    ```