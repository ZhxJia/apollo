## 初始化
### 参数文件结构
此文件中包含障碍物检测跟踪的功能组件：
由camera/app/obstacle_camera_perception.cc中的初始化函数调用此处文件中的初始化

- Plugin:                    name:
- detector          ->       YoloObstacleDetector   
- postprocessor     ->       LocationRefinerObstaclePostprocessor
- tracker           ->       OMTObstacleTracker
- transformer       ->       MultiCueObstacleTransformer

每一个文件夹的初始化参数调用包含两个主要文件
> `*.proto`
> `config.pt`:(位于`production/data/perception/camera/models/`)
>> 上一级配置文件位于`production/conf/perception/camera/obstacle.pt`

### 函数初始化调用部分：
由`perception/camera/lib/interface/*.h`下的各头文件给出各类的一般性定义：包含结构体和纯虚函数。

- 配置文件路径的相关信息由结构体`struct BaseInitOptions {...}`的派生给出：包含`root_dir,conf_file,gpu_id`；
- 通过`name`实例化具体各`Plugin`纯虚函数（通过工厂模式），得到各`Plugin`的功能对象：
    >  YoloObstacleDetector、LocationRefinerObstaclePostprocessor、OMTObstacleTracker、MultiCueObstacleTransformer
- 调用对应的::Init()初始化函数并传入option