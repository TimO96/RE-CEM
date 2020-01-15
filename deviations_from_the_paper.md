- Fixed 'bug' in aen_CEM.py line 131 self.Loss_AE_Dist_s = self.gamma*tf.square(tf.norm(self.AE(self.delta_img)-self.delta_img_s))
- Converted Tensorflow/Keras implementation to Pytorch, hereby altering the session graph to Pytorch
- Instead of max between -kappa and the loss do the max between 0 and loss+kappa


- To be added...
