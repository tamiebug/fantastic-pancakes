#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <vector>

using namespace tensorflow;

REGISTER_OP("RoiPoolingLayer")
	.Attr("T: numbertype")	
	.Input("proposal_regions : T")
	.Input("feature_map: T")
	.Attr("pooled_height : int")
	.Attr("pooled_width : int")
	.Attr("feature_stride : int")
	.Output("outputs : T");

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

template <typename T>
class RoiPoolingLayer : public OpKernel {
	public:
		explicit RoiPoolingLayer(OpKernelConstruction *context) : OpKernel(context) 
		{
				OP_REQUIRES_OK(context, context->GetAttr("pooled_height", &pooled_height));
				OP_REQUIRES_OK(context, context->GetAttr("pooled_width", &pooled_width));
	
		}

		
		void Compute(OpKernelContext *context) override {
				const Tensor& proposal_regions = context->input(0);
				const Tensor& features = context->input(1);
							const int roi_dim = proposal_regions.dim_size(0);

				// Creating an empty output with the proper shape
				Tensor *output_tensor = NULL;
				TensorShape output_shape = {roi_dim,pooled_height,pooled_width};
				OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,\
																&output_tensor));
				
		}
	private:
		int feat_h, feat_w, feat_stride, pooled_height, pooled_width;
};


#define REGISTER_KERNEL(type)						\
		REGISTER_KERNEL_BUILDER(					\
						Name("RoiPoolingLayer")		\
						.Device(DEVICE_CPU)			\
						.TypeConstraint<type>("T"),	\
						RoiPoolingLayer<type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#define REGISTER_KERNEL(type)						\
		REGISTER_KERNEL_BUILDER(					\
						Name("RoiPoolingLayer")		\
						.Device(DEVICE_GPU)			\
						.TypeConstraint<type>("T"),	\
						RoiPoolingLayer<type>);		

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
