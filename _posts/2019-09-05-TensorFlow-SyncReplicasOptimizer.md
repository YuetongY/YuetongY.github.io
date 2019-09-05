---
layout: post
title:  "TensorFlow SyncReplicasOptimizer 解读"
subtitle: "对SyncReplicasOptimizer实现的分布式训练的分析"
author: "yuetong"
header-img: "img/ps.png"
date:   2019-09-05
tags:
    - TensorFlow
    - 深度学习
    - 分布式训练
    - 参数服务器
---

### 背景
在CNN等深度神经网络的分布式训练中，比较常用的一种训练模式是（同步的）数据并行，也就是各个计算设备分别根据各自获得的batch，前向计算获得损失，进而反向传播计算梯度。待所有计算设备完成梯度计算之后，对梯度进行平均，利用平均梯度对模型进行更新。

上面这种模式只是一种逻辑架构，具体实现上，可以使用参数服务器（PS）的形式实现。在参数服务器架构中，计算设备被划分为参数服务器(ps)和worker。对于ps，顾名思义，主要是对模型的参数进行存储；而对于worker，主要的工作是完成前向和反向传播这类计算密集的运算。

在（同步的）参数服务器架构中，整个流程可以分为以下几个步骤：
1. worker从ps把模型参数pull到本地的memory中
2. worker利用模型参数完成前向计算，再完成反向传播，得到模型参数的梯度
3. worker将模型参数的梯度push至参数服务器，参数服务器收到所有worker的梯度之后，计算平均梯度，并对模型参数进行更新

之所以想到查看`SyncReplicasOptimizer`的实现，是因为我在实验中，用到了这个Optimizer，最初看它的名字，想当然的认为它是一个同步的Optimizer，但是实际使用发现如果参数设置不当，可能会出现各个worker计算进度不一致，也就是没有真正的同步。另外，在实验中发现在我的实验环境下，PS架构 + `sync_replicas_optimizer`不能完全发挥集群的性能，可能需要了解sync_replicas_optimizer的工作原理才能更好的发现问题所在。
 
 下面的介绍是我个人查看`tf.train.Sync_replicas_optimizer`实现部分的理解。
 
### SyncReplicasOptimizer解析
#### 文档说明
在TensorFlow r1.14版本的[文档](https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer)中，这个Optimizer提供的API已经被弃用了，在新的API中，如果希望在分布式环境中实现同步的训练，是通过配置[Distribution Strategies](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/distribute)来实现的。
 > This class is deprecated. For synchrononous training, please use [Distribution Strategies](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/distribute).
 
 在文档中，给出了这个优化器的“同步”训练方案，如下：
 > - For the Parameter Server job:
	1. An accumulator is created for each variable, and each replica pushes the gradients into the accumulators instead of directly applying them to the variables.
	2. Each accumulator averages once enough gradients (replicas_to_aggregate) have been accumulated.
	3. Apply the averaged gradients to the variables.
	4. Only after all variables have been updated, increment the global step.
	5. Only after step 4, pushes global_step in the token_queue, once for each worker replica. The workers can now fetch the global step, use it to update its local_step variable and start the next batch. Please note that some workers can consume multiple minibatches, while some may not consume even one. This is because each worker fetches minibatches as long as a token exists. If one worker is stuck for some reason and does not consume a token, another worker can use it.
>- For the replicas:
	> 1. Start a step: fetch variables and compute gradients.
	> 2. Once the gradients have been computed, push them into gradient accumulators. Each accumulator will check the staleness and drop the stale.
	> 3. After pushing all the gradients, dequeue an updated value of global_step from the token queue and record that step to its local_step variable. Note that this is effectively a barrier.
	> 4. Start the next batch.
 
 在文档中，Parameter Server job可以简单理解为参数服务器，而Replicas可以简单理解为各个worker，或者是各个GPU。可以从文档中得到这两个信息：
 - 参数服务器是通过为每一个variable建立一个`accumulator`数据结构，进而对来自不同worker的梯度进行管理（平均）的
 - `token_queue`这个数据结构被用来保持各个worker的进度（一致）
 
 文档中的叙述展示了这个Optimizer的执行逻辑，但是对于TensorFlow而言，它是通过计算图完成计算的，上面的流程如何在计算图中得到体现，就需要查看`sync_replicas_optimizer`的实现部分。
#### 源码部分
##### `__init__`
`sync_replicas_optimizer`是一个wrapper optimizer，也就是这个Optimizer对其他Optimizer进行包装，完成worker间梯度同步的工作，而实际梯度的计算则是交由被包装的Optimizer来完成的。这一点可以通过构造函数发现：
 
```python
def __init__(self,
               opt,
               replicas_to_aggregate,
               total_num_replicas=None,
               variable_averages=None,
               variables_to_average=None,
               use_locking=False,
               name="sync_replicas"):
               if total_num_replicas is None:
				    total_num_replicas = replicas_to_aggregate
			    super(SyncReplicasOptimizer, self).__init__(use_locking, name)
			    logging.info(
			        "SyncReplicasV2: replicas_to_aggregate=%s; total_num_replicas=%s",
			        replicas_to_aggregate, total_num_replicas)
			    self._opt = opt
			    self._replicas_to_aggregate = replicas_to_aggregate
			    self._gradients_applied = False
			    self._variable_averages = variable_averages
			    self._variables_to_average = variables_to_average
			    self._total_num_replicas = total_num_replicas
			    self._tokens_per_step = max(total_num_replicas, replicas_to_aggregate)
			    self._global_step = None
			    self._sync_token_queue = None
			    self._chief_queue_runner = None
			    self._accumulator_list = []
```
 
在构造函数中，可以传入这么几个参数，`opt`,`replicas_to_aggregate`,`total_num_replicas`,`variable_averages`,`variables_to_average`,`use_locking`,`name`。各个参数具体的含义可以参考[官方文档](https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer#__init__)给出的解释。需要说明的有几个点，`opt`这个参数体现了wrapper的思想，它是一个`Optimizer`类的一个对象，通过它来真正完成梯度的计算。`total_num_replicas`和`replicas_to_aggregate`这两个参数会被用来控制同步的进度，前者表示**集群中worker的数目**，而后者表示**ps完成一次参数更新所需要收集的梯度数目**。在同步的训练模式下，它们两个应当是相等的，但是事实上，即使它们俩相等，也有可能导致不同步的现象出现，这与其他参数(`token_num`)的设置有关系。

如果前者比后者大，逻辑上意味着ps在一轮迭代中，只需要收集到部分梯度，就利用这些梯度计算平均梯度，进而对参数更新，这可能意味着这轮迭代中，计算较慢的那个节点的梯度会被丢弃*（只是一种猜测，还未实验验证）*。这两个参数与之前提到的`token_queue`的设置有关系，在后面的源码介绍将会更加详细的说明。
 
在构造函数中，还为这个Optimizer定义了一些新的数据结构，就不详细介绍了，读完整个执行流程之后，它们的功能应该就会比较清楚。

##### `compute_gradients`
下面来看`compute_gradients`的实现。
 
```python
   def compute_gradients(self, *args, **kwargs):
	   return self._opt.compute_gradients(*args, **kwargs)
```
 
 可以从源码中验证wrapper的思想，梯度的计算是通过被包装的Optimizer来实现的。

##### `apply_gradients`
这个方法将会构造模型参数更新部分的计算图，最后将返回一个`train_op`。`train_op`是通常训练过程中，client为session的fetches提供的参数之一，也就是这个Operation被执行之后，模型的参数将会完成更新，并开始下一个batch的训练。那么这也就意味着，这个方法中涉及到的计算图将会实现说明文档中的训练逻辑。

```python
def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")

    if global_step is None:
      raise ValueError("Global step is required to check staleness")

    self._global_step = global_step
    train_ops = []  # 所有的train_op，也就是Accumulator的apply_gradients，实际上可以指push的过程
    aggregated_grad = []  # 平均后的梯度
    var_list = []  # 变量存放的列表
    ...
```

上面的代码负责验证传入方法的参数的合法性，并创建一些供方法内部使用的数据结构。传入的参数必须包含有`global_step`，这在其他Optimizer可能不是一个必须的参数，在这里，它为必须的，因为它需要被用在检查stale gradiets的地方。

```python
    local_anchor = control_flow_ops.no_op()
    with ops.colocate_with(local_anchor):
      self._local_step = variable_scope.variable(
          initial_value=0,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          dtype=global_step.dtype.base_dtype,
          name="sync_rep_local_step")
      # _local_step的初始化Operation
      self.local_step_init_op = state_ops.assign(self._local_step, global_step)
      chief_init_ops = [self.local_step_init_op]
      self.ready_for_local_init_op = variables.report_uninitialized_variables(
        variables.global_variables())
```

这一部分首先在计算图上创建了一个空的Operation，之后创建了一个类的成员变量`_local_step`，用来记录这个worker的计算进度。用到`ops.colocate_with()`的原因是在于，在一般情况下，PS架构会通过device function等机制，将这类Operation放在worker所处的device上，而将Variables这类特殊的Operation放在PS上。`local_anchor`是一个一般的Operation，因此它会被PS架构分配至worker所处的device，这样就能确保创建的`_local_step`变量也能够放在worker上。

```python
    with ops.name_scope(None, self._name):  # 在计算图中创建以这个优化器为名称的name_scope
      for grad, var in grads_and_vars:  # 遍历计算好的每一份梯度
        var_list.append(var)  # 将variables按顺序保存在一个列表里
        with ops.device(var.device):  # 下面定义的计算图的部分将与variables放在相同的device上。可以理解为都被放在PS上
          # Dense gradients.
          if grad is None:
            aggregated_grad.append(None)  # pass-through.
            continue
          elif isinstance(grad, ops.Tensor):
            # 为每一份梯度创建一个ConditionalAccumulator
            grad_accum = data_flow_ops.ConditionalAccumulator(
                grad.dtype,
                shape=var.get_shape(),
                shared_name=var.name + "/grad_accum")
            # 在train_ops这个集合中放入对应accumulator的apply_grad operation
            # 注意，这个apply_grad Operation传入了一个local_step参数，这个操Operation是把梯度放到accumulator中，如果local_step落后于global_step，这个Operation会自动终止，也就是梯度不会放入accumulator
            train_ops.append(grad_accum.apply_grad(
                grad, local_step=self._local_step))
            # aggregrated_grad中有所有平均之后的梯度，它是从accumulator中执行take_grad的Operation
            # take_grad()中有一个参数，num_required，如果说accumulator中的梯度数量少于num_required，它将会阻塞。
            aggregated_grad.append(grad_accum.take_grad(
                self._replicas_to_aggregate))
          else:
		    # 稀疏梯度的处理，与上面类似
            if not isinstance(grad, ops.IndexedSlices):
              raise ValueError("Unknown grad type!")
            grad_accum = data_flow_ops.SparseConditionalAccumulator(
                grad.dtype, shape=(), shared_name=var.name + "/grad_accum")
            train_ops.append(grad_accum.apply_indexed_slices_grad(
                grad, local_step=self._local_step))
            aggregated_grad.append(grad_accum.take_indexed_slices_grad(
                self._replicas_to_aggregate))

          self._accumulator_list.append((grad_accum, var.device))
      aggregated_grads_and_vars = zip(aggregated_grad, var_list)  # 最终的(平均梯度，参数)列表
```

上面的代码给出的是进行梯度聚合，最终获得对应每一个参数的平均梯度的过程。在上面这个过程中，分别为每一个Variable创建了一个`ConditionalAccumulator`，`ConditionalAccumulator`是被多个Session所共享的，用于维护在这个time step下来自不同worker的梯度。ConditionalAccumulator内部实际上维护了一个time step，记录了当前集群训练的进度。在Accumulator中提供了`apply_grad()`方法，这个方法需要将`local_step`作为一个参数传入，返回一个Operation，如果`local_step`小于内部的time step，那这个Operation就不会被执行。而`take_grad()`方法在Accumulator内部收集的gradient数目少于`num_required`参数指出的数目时，就会阻塞，它被调用完之后，Accumulator内部的gradient计数器会清零，同时内部的time step会递增1个单位。

```python
      # sync_op will be assigned to the same device as the global step.
      with ops.device(global_step.device), ops.name_scope(""):
        update_op = self._opt.apply_gradients(aggregated_grads_and_vars,
                                              global_step)  # 真正完成参数更新的地方，执行之后，global_step会+1

      # Create token queue.
      with ops.device(global_step.device), ops.name_scope(""):
        sync_token_queue = (
            data_flow_ops.FIFOQueue(-1,
                                    global_step.dtype.base_dtype,
                                    shapes=(),
                                    name="sync_token_q",
                                    shared_name="sync_token_q"))
        self._sync_token_queue = sync_token_queue

        # dummy_queue is passed to the queue runner. Don't use the real queues
        # because the queue runner doesn't automatically reopen it once it
        # closed queues in PS devices.
        dummy_queue = (
            data_flow_ops.FIFOQueue(1,
                                    types_pb2.DT_INT32,
                                    shapes=(),
                                    name="dummy_queue",
                                    shared_name="dummy_queue"))

      with ops.device(global_step.device), ops.name_scope(""):
        # Replicas have to wait until they can get a token from the token queue.
        with ops.control_dependencies(train_ops):
          token = sync_token_queue.dequeue()  # 这个Operation只依赖与train_ops，而不依赖于update_op
        train_op = state_ops.assign(self._local_step, token)  # 更新_local_step的值为集群所要求的step

        with ops.control_dependencies([update_op]):
          # Sync_op needs to insert tokens to the token queue at the end of the
          # step so the replicas can fetch them to start the next step.
          tokens = array_ops.fill([self._tokens_per_step], global_step)
          sync_op = sync_token_queue.enqueue_many((tokens,))  # 将新的toekns放入队列中，tokens是新的global_step

        if self._variable_averages is not None:
          with ops.control_dependencies([sync_op]), ops.name_scope(""):
            sync_op = self._variable_averages.apply(
                self._variables_to_average)

        self._chief_queue_runner = queue_runner.QueueRunner(dummy_queue,
                                                            [sync_op])
      for accum, dev in self._accumulator_list:
        with ops.device(dev):
          chief_init_ops.append(
              accum.set_global_step(
                  global_step, name="SetGlobalStep"))
      self.chief_init_op = control_flow_ops.group(*(chief_init_ops))
      self._gradients_applied = True
      return train_op
```

上面的流程说明了如何通过`token_queue`控制各个worker的计算进度。在上面代码给出的流程中，我们可以将token看做计算集群当前要求各个worker完成的任务，如果worker获得了token，那么代表着这个worker领取了token表示的`global_step`的任务。`token_queue`的更新是需要等待真实的Optimizer完成了参数的更新之后，才会向token queue中放入一定数目的新的token。

有一点需要注意，取token这个Operation只依赖于`train_ops`，而`train_ops`表示着在Accumulator中的apply_gradients的Operation，可以简单的把`train_ops`看做push的过程。这意味着，当worker完成了一轮梯度计算之后，如果`token_queue`中还有剩余的token，那么它将会领取一个剩余的token，利用下一个batch的数据开始一轮相同的迭代。

至此，`sync_replicas_optimizer`中最重要的，利用平均后的梯度更新模型参数的方法`apply_gradients()`的执行流程，应该已经比较清楚了，总结一下，比较重要的是有以下几个点：
1. ConditionalAccumulator的梯度聚合
ConditionalAccumulator可以聚合那些合适的梯度，stale的梯度值会被自动抛弃，且直到收集到足够的梯度之后，才能获得平均梯度。
2. Token控制训练进度
在[官方文档](https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer#get_init_tokens_op)中，对Token提到的关键作用在于，当worker数目少于每一轮迭代需要收集的梯度(`total_num_replicas`  <  ` replicas_to_aggregate`)
时，训练进程能够正常迭代下去。但是除了文档提到的这个特性之外，在整个训练过程开始之前，初始的token数目将会影响到各个worker的训练进度。下面将先看一下初始token数目的设置，再分析token数目对训练进度产生影响的原因。

##### `get_init_tokens_op`
```python
  def get_init_tokens_op(self, num_tokens=-1):
    if self._gradients_applied is False:
      raise ValueError(
          "get_init_tokens_op() should be called after apply_gradients().")

    tokens_needed = self._replicas_to_aggregate - self._total_num_replicas
    if num_tokens == -1:
      num_tokens = self._replicas_to_aggregate
    elif num_tokens < tokens_needed:
      raise ValueError(
          "Too few tokens to finish the first step: %d (given) vs %d (needed)" %
          (num_tokens, tokens_needed))

    if num_tokens > 0:
      with ops.device(self._global_step.device), ops.name_scope(""):
        tokens = array_ops.fill([num_tokens], self._global_step)
        init_tokens = self._sync_token_queue.enqueue_many((tokens,))
    else:
      init_tokens = control_flow_ops.no_op(name="no_init_tokens")

    return init_tokens

  def make_session_run_hook(self, is_chief, num_tokens=-1):
    """Creates a hook to handle SyncReplicasHook ops such as initialization."""
    return _SyncReplicasOptimizerHook(self, is_chief, num_tokens)
```

这部分的代码给出了在训练开始之前，向token_queue放入多少token的执行流程。这一部分的代码会通过hook的方式，在session创建之后，对模型参数初始化时执行。

在默认情况下，如果不人为指定参数，传入的参数`num_tokens`为`-1`，此时将会在`token_queue`中放入`replicas_to_aggregate`个token。在实际实验过程中，这将会导致一些不同步的问题，在issues [#11753](https://github.com/tensorflow/tensorflow/issues/11753#issuecomment-377855509)中提到了解决方法，是将初始token数目设置为0即可。我通过实验也验证了这个方法的有效性。

上面的`apply_gradients`的执行流程，可以解释这个现象发生的原因。因为当worker完成了向参数服务器 push 梯度的过程之后，就会申请dequeue，如果此时`token_queue`中有token，那么worker就能取新的训练数据，依然利用刚才的模型参数再计算一轮梯度。实际上，如果各个worker算力差别很大，是可能出现K-batch-sync SGD这种训练模式的，下图很好的说明了这一点。下图的设置可以理解为`total_num_replicas=3`,`replicas_to_aggregate=2`,的情况。L2的计算速度相较于另外两个worker快很多，由于它本地迭代一轮之后，可以从queue中取出一个token，开始新的一轮计算。本地迭代两轮之后，另外两个worker还未迭代完一轮，这时，参数服务器将会利用L2的两次梯度进行平均，对参数更新之后，`global_step`递增1，开始全局的新一轮计算。
![@K-batch-sync SGD](/img/1567651943862.png)
### 参考资料
1. [tf.train.SyncReplicasOptimizer no synchronization among workers #11753](https://github.com/tensorflow/tensorflow/issues/11753)
2. [Synchronous distributed tensorflow training doesn't synchronize among workers #9596](https://github.com/tensorflow/tensorflow/issues/9596)
3. [tf.train.SyncReplicasOptimizer](tf.train.SyncReplicasOptimizer)
4. [Optimizer in Tensorflow](https://zhuanlan.zhihu.com/p/40342278)
5. [Slow and Stale Gradients Can Win the Race: Error-Runtime Trade-offs in Distributed SGD](https://arxiv.org/abs/1803.01113)


