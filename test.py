import tensorflow as tf

from tensorflow.python.ops import array_ops


sess=tf.Session()

l=[]
for j in xrange(5):
    pos=[i for i in xrange(10)]
    l.append(pos)

qiu=tf.reshape(l,[5,10])

j=[1,2,3]
j=tf.reshape(j,[3,1])
q=tf.ones([4,1],dtype=tf.int32)
q=tf.reshape(q,[1,4])
#y=tf.matmul(j,q)
y=j*q
y=tf.reshape(y,[3,4,1])

n=[1,2,3,4]
n=tf.reshape(n,[1,4,1])

m=y*n
print (m.get_shape())
print (sess.run([m]))

qiu=[1,3,2]
qiu=tf.reshape(qiu,[1,3])
qiu=tf.Tensor.eval(qiu)

for one in qiu:
    print (one)









