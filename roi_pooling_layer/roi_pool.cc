#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("RoiPoolingLayer")
	.Attr("T: numbertype")
	.Input("proposal_regions : T")
	.Input("feature_map: T")
	.Output("outputs : T")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c){
			c->set_output(0, c->input(1));
			return Status::OK();
			});

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

template <typename T>
class RoiPoolingLayer : public OpKernel {
	public:
		explicit RoiPoolingLayer(OpKernelConstruction *context) : OpKernel(context) {}
		
		void Compute(OpKernelContext *context) override {
				const Tensor& proposal_regions = context->input(0);
				const Tensor& features = context->input(1);
				
				// Creating a garbage output tensor for testing.  Proof of concept,
				// should return the identity
				Tensor *output_tensor = NULL;
				OP_REQUIRES_OK(context, context->allocate_output(0, features.shape(),
																&output_tensor));
				// Creating a hard copy since context->input(A) objects are const.
				*output_tensor = context->input(1);	
				}
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
