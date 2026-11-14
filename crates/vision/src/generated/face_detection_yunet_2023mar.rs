// Generated from ONNX "/var/folders/vq/1wh9t5d5487bq7cf3y400n4r0000gn/T/tmp.F97HjEf2AT/face_detection_yunet_2023mar.onnx" by burn-onnx
extern crate alloc;
use burn::prelude::*;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;


#[derive(Module, Debug)]
pub struct Model {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    maxpool2d1: MaxPool2d,
    conv2d4: Conv2d,
    conv2d5: Conv2d,
    conv2d6: Conv2d,
    conv2d7: Conv2d,
    conv2d8: Conv2d,
    conv2d9: Conv2d,
    conv2d10: Conv2d,
    conv2d11: Conv2d,
    maxpool2d2: MaxPool2d,
    conv2d12: Conv2d,
    conv2d13: Conv2d,
    conv2d14: Conv2d,
    conv2d15: Conv2d,
    maxpool2d3: MaxPool2d,
    conv2d16: Conv2d,
    conv2d17: Conv2d,
    conv2d18: Conv2d,
    conv2d19: Conv2d,
    maxpool2d4: MaxPool2d,
    conv2d20: Conv2d,
    conv2d21: Conv2d,
    conv2d22: Conv2d,
    conv2d23: Conv2d,
    conv2d24: Conv2d,
    conv2d25: Conv2d,
    resize1: burn::nn::interpolate::Interpolate2d,
    conv2d26: Conv2d,
    conv2d27: Conv2d,
    resize2: burn::nn::interpolate::Interpolate2d,
    conv2d28: Conv2d,
    conv2d29: Conv2d,
    conv2d30: Conv2d,
    conv2d31: Conv2d,
    conv2d32: Conv2d,
    conv2d33: Conv2d,
    conv2d34: Conv2d,
    conv2d35: Conv2d,
    conv2d36: Conv2d,
    conv2d37: Conv2d,
    conv2d38: Conv2d,
    conv2d39: Conv2d,
    conv2d40: Conv2d,
    conv2d41: Conv2d,
    conv2d42: Conv2d,
    conv2d43: Conv2d,
    conv2d44: Conv2d,
    conv2d45: Conv2d,
    conv2d46: Conv2d,
    conv2d47: Conv2d,
    conv2d48: Conv2d,
    conv2d49: Conv2d,
    conv2d50: Conv2d,
    conv2d51: Conv2d,
    conv2d52: Conv2d,
    conv2d53: Conv2d,
    #[module(skip)]
    device: Device,
}

impl Model {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 16], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([16, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([16, 16], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(16)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let conv2d4 = Conv2dConfig::new([16, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([16, 16], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(16)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([16, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(32)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([32, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(32)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let maxpool2d2 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let conv2d12 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let maxpool2d3 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let conv2d16 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let maxpool2d4 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let conv2d20 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let resize1 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d26 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let resize2 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d28 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([64, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([1, 1], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([64, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([1, 1], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([64, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([1, 1], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([64, 4], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([4, 4], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([64, 4], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([4, 4], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([64, 4], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([4, 4], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([64, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([1, 1], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([64, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([1, 1], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([64, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([1, 1], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([64, 10], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([10, 10], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(10)
            .with_bias(true)
            .init(device);
        let conv2d50 = Conv2dConfig::new([64, 10], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([10, 10], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(10)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([64, 10], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([10, 10], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(10)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            conv2d2,
            conv2d3,
            maxpool2d1,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            maxpool2d2,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            maxpool2d3,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            maxpool2d4,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            conv2d25,
            resize1,
            conv2d26,
            conv2d27,
            resize2,
            conv2d28,
            conv2d29,
            conv2d30,
            conv2d31,
            conv2d32,
            conv2d33,
            conv2d34,
            conv2d35,
            conv2d36,
            conv2d37,
            conv2d38,
            conv2d39,
            conv2d40,
            conv2d41,
            conv2d42,
            conv2d43,
            conv2d44,
            conv2d45,
            conv2d46,
            conv2d47,
            conv2d48,
            conv2d49,
            conv2d50,
            conv2d51,
            conv2d52,
            conv2d53,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> (
        Tensor<3>,
        Tensor<3>,
        Tensor<3>,
        Tensor<3>,
        Tensor<3>,
        Tensor<3>,
        Tensor<3>,
        Tensor<3>,
        Tensor<3>,
        Tensor<3>,
        Tensor<3>,
        Tensor<3>,
    ) {
        let conv2d1_out1 = self.conv2d1.forward(input);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let conv2d3_out1 = self.conv2d3.forward(conv2d2_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu2_out1);
        let conv2d4_out1 = self.conv2d4.forward(maxpool2d1_out1);
        let conv2d5_out1 = self.conv2d5.forward(conv2d4_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu3_out1);
        let conv2d7_out1 = self.conv2d7.forward(conv2d6_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu4_out1);
        let conv2d9_out1 = self.conv2d9.forward(conv2d8_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu5_out1);
        let conv2d11_out1 = self.conv2d11.forward(conv2d10_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let maxpool2d2_out1 = self.maxpool2d2.forward(relu6_out1);
        let conv2d12_out1 = self.conv2d12.forward(maxpool2d2_out1);
        let conv2d13_out1 = self.conv2d13.forward(conv2d12_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu7_out1);
        let conv2d15_out1 = self.conv2d15.forward(conv2d14_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let maxpool2d3_out1 = self.maxpool2d3.forward(relu8_out1.clone());
        let conv2d16_out1 = self.conv2d16.forward(maxpool2d3_out1);
        let conv2d17_out1 = self.conv2d17.forward(conv2d16_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv2d17_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu9_out1);
        let conv2d19_out1 = self.conv2d19.forward(conv2d18_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let maxpool2d4_out1 = self.maxpool2d4.forward(relu10_out1.clone());
        let conv2d20_out1 = self.conv2d20.forward(maxpool2d4_out1);
        let conv2d21_out1 = self.conv2d21.forward(conv2d20_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv2d21_out1);
        let conv2d22_out1 = self.conv2d22.forward(relu11_out1);
        let conv2d23_out1 = self.conv2d23.forward(conv2d22_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d23_out1);
        let conv2d24_out1 = self.conv2d24.forward(relu12_out1);
        let conv2d25_out1 = self.conv2d25.forward(conv2d24_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d25_out1);
        let resize1_out1 = self.resize1.forward(relu13_out1.clone());
        let add1_out1 = relu10_out1.add(resize1_out1);
        let conv2d26_out1 = self.conv2d26.forward(add1_out1);
        let conv2d27_out1 = self.conv2d27.forward(conv2d26_out1);
        let relu14_out1 = burn::tensor::activation::relu(conv2d27_out1);
        let resize2_out1 = self.resize2.forward(relu14_out1.clone());
        let add2_out1 = relu8_out1.add(resize2_out1);
        let conv2d28_out1 = self.conv2d28.forward(add2_out1);
        let conv2d29_out1 = self.conv2d29.forward(conv2d28_out1);
        let relu15_out1 = burn::tensor::activation::relu(conv2d29_out1);
        let conv2d30_out1 = self.conv2d30.forward(relu15_out1.clone());
        let conv2d31_out1 = self.conv2d31.forward(conv2d30_out1);
        let conv2d32_out1 = self.conv2d32.forward(relu14_out1.clone());
        let conv2d33_out1 = self.conv2d33.forward(conv2d32_out1);
        let conv2d34_out1 = self.conv2d34.forward(relu13_out1.clone());
        let conv2d35_out1 = self.conv2d35.forward(conv2d34_out1);
        let conv2d36_out1 = self.conv2d36.forward(relu15_out1.clone());
        let conv2d37_out1 = self.conv2d37.forward(conv2d36_out1);
        let conv2d38_out1 = self.conv2d38.forward(relu14_out1.clone());
        let conv2d39_out1 = self.conv2d39.forward(conv2d38_out1);
        let conv2d40_out1 = self.conv2d40.forward(relu13_out1.clone());
        let conv2d41_out1 = self.conv2d41.forward(conv2d40_out1);
        let conv2d42_out1 = self.conv2d42.forward(relu15_out1.clone());
        let conv2d43_out1 = self.conv2d43.forward(conv2d42_out1);
        let conv2d44_out1 = self.conv2d44.forward(relu14_out1.clone());
        let conv2d45_out1 = self.conv2d45.forward(conv2d44_out1);
        let conv2d46_out1 = self.conv2d46.forward(relu13_out1.clone());
        let conv2d47_out1 = self.conv2d47.forward(conv2d46_out1);
        let conv2d48_out1 = self.conv2d48.forward(relu15_out1);
        let conv2d49_out1 = self.conv2d49.forward(conv2d48_out1);
        let conv2d50_out1 = self.conv2d50.forward(relu14_out1);
        let conv2d51_out1 = self.conv2d51.forward(conv2d50_out1);
        let conv2d52_out1 = self.conv2d52.forward(relu13_out1);
        let conv2d53_out1 = self.conv2d53.forward(conv2d52_out1);
        let transpose1_out1 = conv2d31_out1.permute([0, 2, 3, 1]);
        let reshape1_out1 = transpose1_out1.reshape([1, -1, 1]);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(reshape1_out1);
        let transpose2_out1 = conv2d33_out1.permute([0, 2, 3, 1]);
        let reshape2_out1 = transpose2_out1.reshape([1, -1, 1]);
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(reshape2_out1);
        let transpose3_out1 = conv2d35_out1.permute([0, 2, 3, 1]);
        let reshape3_out1 = transpose3_out1.reshape([1, -1, 1]);
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(reshape3_out1);
        let transpose4_out1 = conv2d43_out1.permute([0, 2, 3, 1]);
        let reshape4_out1 = transpose4_out1.reshape([1, -1, 1]);
        let sigmoid4_out1 = burn::tensor::activation::sigmoid(reshape4_out1);
        let transpose5_out1 = conv2d45_out1.permute([0, 2, 3, 1]);
        let reshape5_out1 = transpose5_out1.reshape([1, -1, 1]);
        let sigmoid5_out1 = burn::tensor::activation::sigmoid(reshape5_out1);
        let transpose6_out1 = conv2d47_out1.permute([0, 2, 3, 1]);
        let reshape6_out1 = transpose6_out1.reshape([1, -1, 1]);
        let sigmoid6_out1 = burn::tensor::activation::sigmoid(reshape6_out1);
        let transpose7_out1 = conv2d37_out1.permute([0, 2, 3, 1]);
        let reshape7_out1 = transpose7_out1.reshape([1, -1, 4]);
        let transpose8_out1 = conv2d39_out1.permute([0, 2, 3, 1]);
        let reshape8_out1 = transpose8_out1.reshape([1, -1, 4]);
        let transpose9_out1 = conv2d41_out1.permute([0, 2, 3, 1]);
        let reshape9_out1 = transpose9_out1.reshape([1, -1, 4]);
        let transpose10_out1 = conv2d49_out1.permute([0, 2, 3, 1]);
        let reshape10_out1 = transpose10_out1.reshape([1, -1, 10]);
        let transpose11_out1 = conv2d51_out1.permute([0, 2, 3, 1]);
        let reshape11_out1 = transpose11_out1.reshape([1, -1, 10]);
        let transpose12_out1 = conv2d53_out1.permute([0, 2, 3, 1]);
        let reshape12_out1 = transpose12_out1.reshape([1, -1, 10]);
        (
            sigmoid1_out1,
            sigmoid2_out1,
            sigmoid3_out1,
            sigmoid4_out1,
            sigmoid5_out1,
            sigmoid6_out1,
            reshape7_out1,
            reshape8_out1,
            reshape9_out1,
            reshape10_out1,
            reshape11_out1,
            reshape12_out1,
        )
    }
}
