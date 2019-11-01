# Time Series Classification with SGAN and UDA

### Dataset

n_labels = 5 (0 ~ 4)<br>

n_trainset(data/trainset.csv) = 17235<br>
approximately 3.2k samples for each label

n_validset(data/validset.csv) = 2154<br>
approximately 400 samples for each label

n_testset(data/testset.csv) = same as validset

number of unlabeled dataset = 62168

### Series

ex. data/train_series.npy<br>
shape = (17235, 4, 8)<br>

sample:<br>
array([[ 0.23853505, -0.99826982, -0.9920425 , -0.99039374, -0.99140969, -0.99407501, -0.92736046, -0.97322798],<br>
       [ 0.1004917 , -0.99649176, -0.99250318, -0.99032046, -0.94801762, -0.99432381, -0.93127861, -0.97645392],<br>
       [-0.13615405, -0.99554222, -0.99206423, -0.99023892, -0.97577093,
        -0.99379309, -0.93288488, -0.97875748],<br>
       [-0.35307932, -0.99223006, -0.99163833, -0.98970589, -0.95088106,
        -0.99333554, -0.91714334, -0.98305426]])