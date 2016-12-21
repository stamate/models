# models
various machine learning models

### To use any of the models please import as follows

```python
from models import SIMPLE, RCL, ResNet, new_resnet

model = RCL(layers=3, conv_shape=(256, 9), timesteps=3, time_conv_shape=(256, 9), input_shape=(1024, 124), targets=5, optimizer='adam', pool=2, leak=0.3, drop=0.5, time_leak=0.2, time_drop=0.1)

model.fit(X, y)
```
