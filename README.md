apt update && apt install vim

pip install torch future -i https://pypi.tuna.tsinghua.edu.cn/simple

cp upsample_layer.hpp /opt/caffe/include/caffe/layers/
cp upsample_layer.cpp upsample_layer.cu /opt/caffe/src/caffe/layers/

vim /opt/caffe/src/caffe/proto/caffe.proto
...
//  optional WindowDataParameter window_data_param = 129;
  ...
  optional UpsampleParameter upsample_param = 149;
//}
...

...
//message PReLUParameter {
...
//}
...
message UpsampleParameter{
  optional int32 scale = 1 [default = 1];
}

cd /opt/caffe/build
cmake -DUSE_CUDNN=1 -DUSE_NCCL=0 ..
make -j"$(nproc)"

python darknet2caffe.py voc.cfg  yolov3_tiny.weights yolov3-tiny.prototxt yolov3-tiny.caffemodel
