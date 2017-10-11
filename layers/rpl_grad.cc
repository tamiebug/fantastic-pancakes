#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <vector>
#include <limits>
#include <cfloat>

using namespace tensorflow;

REGISTER_OP("RoiPoolerGrad")
	.Input("features : float")
	.Input("image_attr : float")
	.Input("proposal_regions : float")
	.Input("gradient : float")
	.Attr("pooled_height : int")
	.Attr("pooled_width : int")
	.Attr("feature_stride : int")
	.Output("backprop : float");

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

class RoiPooler : public OpKernel {
private:
    int feat_stride, pooled_w, pooled_h;
public:
    explicit RoiPooler(OpKernelConstruction *context) : OpKernel(context) {
		// Save Attributes
		OP_REQUIRES_OK(context, context->GetAttr("pooled_height", &pooled_h));
		OP_REQUIRES_OK(context, context->GetAttr("pooled_width", &pooled_w));
		OP_REQUIRES_OK(context, context->GetAttr("feature_stride", &feat_stride));
	}
    void Compute(OpKernelContext *context) override
		{
		// Providing pointers to inputs/outputs
		const Tensor* _features = &(context->input(0));
		const Tensor* _image_attr = &(context->input(1));
		const Tensor* _proposal_regions = &(context->input(2));
		const Tensor* _gradient = &(context->input(3));
	
		Tensor* _backprop = NULL;
		Tensor _argmax;
		Tensor _pooled_features;

		/* Saving useful input dimensionality constants.  We make the following assumptions:
		 * features.shape    = (feat_h, feat_w, feat_c)
		 * proposal_regions.shape = (num_rois, {x0, y0, x1, y1}) [In other words, (num_rois,4)]
		 * pooled_features.shape    = (num_rois, pooled_h, pooled_w, feat_c)
		 * argmax.shape    = (num_rois, pooled_h, pooled_w, feat_c, {h,w})
		 *						[in other words, (num_rois, pooled_h, pooled_w, feat_c, 2)]
		 */
		const int num_rois = _proposal_regions->dim_size(0);
		const int feat_w = _features->dim_size(1);
		const int feat_h = _features->dim_size(0);
		const int feat_c = _features->dim_size(2);
		
		// The last dimension of image_attr has the scaling factor we want
		const int scale_factor = (_image_attr->template tensor<float,1>())(2);

		/* Images are resized before being fed into the network.  Hence, the true
		 * feature stride is the inherent feature stride of the underlying classification
		 * network (16 for vgg16/19 for example) times the rescaling done to the image
		 * before being input into that network
		 */
		const float true_feat_stride = float(feat_stride) * float(scale_factor);

		// With these constants established, we can allocate a buffer for the output
		TensorShape _backprop_shape = {feat_h, feat_w, feat_c};
		OP_REQUIRES_OK(context, context->allocate_output(0, _backprop_shape, 
					&_backprop));
		
		// Allocate buffers for temporary tensors needed for backprop calculations
		TensorShape _argmax_shape = {num_rois, pooled_h, pooled_w, feat_c, 2};
		OP_REQUIRES_OK(context, context->allocate_temp(tensorflow::DataType::DT_FLOAT,
													 	_argmax_shape, &_argmax));
		TensorShape _pooled_features_shape = {num_rois, pooled_h, pooled_w, feat_c};
		OP_REQUIRES_OK(context, context->allocate_temp(tensorflow::DataType::DT_FLOAT,
			_pooled_features_shape, &_pooled_features));

		// Create Eigen::Tensor views into each buffer.  They are easier to work with than
		// tensorflow::Tensor objects and will be used in lieu of them for rest of code
		auto features = _features->template tensor<float,3>();
		auto pooled_features = _pooled_features.template tensor<float,4>();
		auto proposal_regions = _proposal_regions->template tensor<float,2>();
		
		auto gradient = _gradient->template tensor<float,4>();
		auto backprop = _backprop->template tensor<float,3>();	
		auto argmax = _argmax.template tensor<float,5>(); 
		
		// We need to initialize the pooled output to a negative value 
		// Not sure of more efficient way, doing it this way for now.
		for(int a=0; a<num_rois; a++){
		    for(int b=0; b<pooled_h; b++){
				for(int c=0; c<pooled_w; c++){
				    for(int d=0; d<feat_c; d++){
						pooled_features(a,b,c,d) = std::numeric_limits<float>::min();
					}
				}
			}
		}

		// Likewise, we need to initialize the gradients to zero
		backprop.setZero();
		/*
		for(int a=0; a<feat_h; a++){
			for(int b=0; b<feat_w; b++){
				for(int c=0; c<feat_c; c++){
					backprop(a,b,c) = 0.;
				}
			}
		}*/
		
		
		for( int n=0 ; n < num_rois ; n++ ) {
		    // Region of interest translated to feature input
			const int roi_w_feat_start = static_cast<int>(
							std::floor(proposal_regions(n,0)/true_feat_stride + .5));
			const int roi_h_feat_start = static_cast<int>(
							std::floor(proposal_regions(n,1)/true_feat_stride + .5));
			const int roi_w_feat_end = static_cast<int>(
							std::ceil(proposal_regions(n,2)/true_feat_stride + .5));
			const int roi_h_feat_end = static_cast<int>(
							std::ceil(proposal_regions(n,3)/true_feat_stride + .5));
			// pw, ph are individual elements of pooled output 
			const float pw_binsize = (roi_w_feat_end - roi_w_feat_start)/
					static_cast<float>(pooled_w);
			const float ph_binsize = (roi_h_feat_end - roi_h_feat_start)/
					static_cast<float>(pooled_h);
            
			// Iterate over elements (b, h, w, c) of the pooled output
			// b = batch, c = channel, h = height, w = width
			for( int c=0; c < feat_c ; c++ ) {
				for( int ph=0 ; ph < pooled_h ; ph++ ) {
				    for( int pw=0 ; pw < pooled_w ; pw++ ) {
						// Check if gradient here is zero.  If so, we can skip computation
						// This check was breaking a unit test, disabled until further notice.
						/*if (std::abs(gradient(n, ph, pw, c)) <= FLT_EPSILON) {
							break;
						}*/
							
						/* Transform ph,pw into h,w region in feature input to be pooled
						 * Make sure to clip dimensions of regions to lie within
						 * the image, e.g. within [0,0] and [feat_h, feat_w]
						 */
						
						int h_0 = static_cast<int>(std::floor( ph * ph_binsize ));
						h_0 = std::min(std::max(h_0 + roi_h_feat_start, 0), feat_h);
						
						int w_0 = static_cast<int>(std::floor( pw * pw_binsize));
						w_0 = std::min(std::max(w_0 + roi_w_feat_start, 0), feat_w);

						int h_f = static_cast<int>(std::ceil( (ph+1) * ph_binsize ));
						h_f = std::min(std::max(h_f + roi_h_feat_start, 0), feat_h);

						int w_f = static_cast<int>(std::ceil( (pw+1) * pw_binsize ));
						w_f = std::min(std::max(w_f + roi_w_feat_start, 0), feat_w);
						

						// Calculate the argmax; This will let us know where the output
						// came from, hence allowing us to backpropagate gradients correctly
						for ( int h=h_0 ; h<h_f ; h++ ) {
						    for ( int w=w_0 ; w<w_f ; w++ ) {
								if ( features(h, w, c) > pooled_features(n, ph, pw, c) ){
								    pooled_features(n, ph, pw, c) = features(h, w, c);
								    argmax(n, ph, pw, c, 0) = h;
								    argmax(n, ph, pw, c, 1) = w;
								}
							}
						}
						
						// Backpropagate outputs to the correct regions
						// We must check for invalid regions where the above for-loop does not
						// execute at all, leaving pooled_features at an invalid, uninitialied value	
						if ((h_f > h_0) && (w_f > w_0)){
							// Assume single batches
							int out_y = argmax(n, ph, pw, c, 0);
							int out_x = argmax(n, ph, pw, c, 1);
							// We do a sum here, but under normal circumstances only one
							// of the channels of gradient will have nonzero values
							backprop(out_y, out_x, c) += gradient(n, ph, pw, c);
						}
					}
				}
			}	
		} 
    }
};
 
REGISTER_KERNEL_BUILDER(\
		Name("RoiPoolerGrad")\
		.Device(DEVICE_CPU)\
		,RoiPooler);
