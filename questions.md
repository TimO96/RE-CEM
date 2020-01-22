# Questions 07/01
- Existing implementation is in Tensorflow, convert to Pytorch? `Yes`
- Standard deviation report for 10-fold cross validation `Can't do fraud anyway`
- Procurement fraud dataset not fully publicly available, same for brain imaging `Brain imaging is`

`The model(s) and all the other files needs to be generate the
results should be in separated`

# Questions 10/01
- Reproduce comparison for LIME and LRP? `NO`
- fMRI data set is downloaded, but we do not know how to continue. Corresponding files from the ABIDE Github need C-PAC which is only available in Python 2. 
- Instruction on how to estimate Pearson matrix as described for the fMRI, generally what to do for the (pre-)processing for the fMRI data. `On the backburner`
- template report `Yes, we'll get it in the mail`
- elaboration on variable new_vars in aen_CEM.py
- perform experiment on different dataset, such as Fashion-MNIST `Yes`
- difference between variables with and without self.adv_img vs self.adv_img_s `Figured it out`
- what model is being trained, if any? what does the attack function supposed to mean exactly? `The delta`
- mnist.h5 for MNISTModel pretrained required? `Can be used, also easy to retrain`
- are we expected to train both models (autoencoder-decoder and MNISTModel) from the start up? if so: optimizer specifications for the MNISTModel are not mentioned in the paper `Possible when changing datasets, not necessary for MNIST`

# Questions 14/01
- Should we als report differences in implementation and paper (e.g. 60.000 training examples vs. 55.000 training examples), or only if the results are then different (or not at all)? `Use what they used, unless it really makes a difference (maybe report both then)`
- Setting the random seed? `Report anything on multiple random runs`
- Using ground truth or prediction to compare changing classes (Section 4.4) `Switch w.r.t. classifier, easy to check metric`
- What is this 10000 doing in the max_nontarget_lab_score? `I don't know, makes no sense check -1, target_lab, scaling`
- Loss after training step? `Could be just wrong for the first grad`
- Structure report? `Explain the algorithm and implementation (+changes), setup, most interseting is discussion + broad interpretation noreal broad implementation this is nice and works okayish bt not usable because continous image f.e. target other scientists write optimization formulas`
- Kappa maximum (eq 4/2) `thing in the appendix is just wrong, in the end it doesnt matter, report if it converges to kappa or zero`

# Questions 17/01
- Skribbl.io?
- Trying to get .996 (batch size 64):
    - 55000 training examples
        ```
        <class 'torch.optim.adam.Adam'> 0.01
        Train: 0.97 test: 0.974 valid: 0.973
        <class 'torch.optim.adam.Adam'> 0.001
        Train: 0.99 test: 0.986 valid: 0.983
        <class 'torch.optim.adam.Adam'> 0.0001
        Train: 0.98 test: 0.978 valid: 0.978
        <class 'torch.optim.sgd.SGD'> 0.01
        Train: 0.78 test: 0.789 valid: 0.788
        <class 'torch.optim.sgd.SGD'> 0.001
        Train: 0.1 test: 0.101 valid: 0.099
        <class 'torch.optim.sgd.SGD'> 0.0001
        Train: 0.11 test: 0.112 valid: 0.108
        <class 'torch.optim.adadelta.Adadelta'> 0.01
        Train: 0.64 test: 0.649 valid: 0.646
        <class 'torch.optim.adadelta.Adadelta'> 0.001
        Train: 0.1 test: 0.101 valid: 0.099
        <class 'torch.optim.adadelta.Adadelta'> 0.0001
        Train: 0.12 test: 0.12 valid: 0.123
        <class 'torch.optim.adagrad.Adagrad'> 0.01
        Train: 0.99 test: 0.991 valid: 0.99
        <class 'torch.optim.adagrad.Adagrad'> 0.001
        Train: 0.95 test: 0.957 valid: 0.96
        <class 'torch.optim.adagrad.Adagrad'> 0.0001
        Train: 0.83 test: 0.84 valid: 0.831
        ```
    - 60000 training examples
        ```
        <class 'torch.optim.adam.Adam'> 0.01
        Train: 0.97 test: 0.976 valid: -1
        <class 'torch.optim.adam.Adam'> 0.001
        Train: 0.99 test: 0.989 valid: -1
        <class 'torch.optim.adam.Adam'> 0.0001
        Train: 0.98 test: 0.98 valid: -1
        <class 'torch.optim.sgd.SGD'> 0.01
        Train: 0.85 test: 0.862 valid: -1
        <class 'torch.optim.sgd.SGD'> 0.001
        Train: 0.1 test: 0.101 valid: -1
        <class 'torch.optim.sgd.SGD'> 0.0001
        Train: 0.11 test: 0.11 valid: -1
        <class 'torch.optim.adadelta.Adadelta'> 0.01
        Train: 0.81 test: 0.823 valid: -1
        <class 'torch.optim.adadelta.Adadelta'> 0.001
        Train: 0.1 test: 0.101 valid: -1
        <class 'torch.optim.adadelta.Adadelta'> 0.0001
        Train: 0.12 test: 0.121 valid: -1
        <class 'torch.optim.adagrad.Adagrad'> 0.01
        Train: 0.99 test: 0.989 valid: -1
        <class 'torch.optim.adagrad.Adagrad'> 0.001
        Train: 0.96 test: 0.963 valid: -1
        <class 'torch.optim.adagrad.Adagrad'> 0.0001
        Train: 0.83 test: 0.844 valid: -1
        ```
# Questions 21/01
- What should be in the ipynb file? `We'll get an example on canvas`
- Why could the addition of the autoencoder (both our trained and their pre-trained) worsen our results? `Scaling`
- Method almost identical to the their method? `Yes`
- PP not working `Scaling`
- Introduction `Include the claims in the paper`

# Questions 24/01
- Thresholding the result images?
- Inits with zeroes works, with images doesn't
