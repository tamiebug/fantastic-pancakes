#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <vector>
#include <limits>

using namespace tensorflow;

REGISTER_OP("RoiPooler")
	.Input("input_matrix : float")
	.Input("image_attr : float")
	.Input("proposal_regions : float")
	.Attr("pooled_height : int")
	.Attr("pooled_width : int")
	.Attr("feature_stride : int")
	.Output("output_matrix: float");

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
		const Tensor *input_layer = &(context->input(0));
		const Tensor *image_attr = &(context->input(1));
		const Tensor *proposal_regions = &(context->input(2));

		Tensor *output_tensor = NULL;

		/* Saving useful input dimensionality constants.  We make the following assumptions:
		 * feature_input.shape    = (feat_h, feat_w, feat_c)
		 * proposal_regions.shape = (num_rois, {x0, y0, x1, y1}) [In other words, (num_rois,4)]
		 * pooled_output.shape    = (num_rois, pooled_h, pooled_w, feat_c)
		 *						[in other words, (num_rois, pooled_h, pooled_w, feat_c, 2)]
		 */
		const int num_rois = proposal_regions->dim_size(0);
		const int feat_w = input_layer->dim_size(1);
		const int feat_h = input_layer->dim_size(0);
		const int feat_c = input_layer->dim_size(2);
		// The last dimension of image_attr has the scaling factor we want
		const int scale_factor = (image_attr->template tensor<float,1>())(2);

		/* Images are resized before being fed into the network.  Hence, the true
		 * feature stride is the inherent feature stride of the underlying classification
		 * network (16 for vgg16/19 for example) times the rescaling done to the image
		 * before being input into that network
		 */
		const float true_feat_stride = float(feat_stride) * float(scale_factor);

		// With these constants established, we can allocate buffers for the outputs
		TensorShape output_tensor_shape = {num_rois, pooled_h, pooled_w, feat_c};
		OP_REQUIRES_OK(context, context->allocate_output(0,output_tensor_shape, &output_tensor));
		
		// Create Eigen::Tensor views into each buffer.  They are easier to work with than
		// tensorflow::Tensor objects and will be used in lieu of them for rest of code
		auto feat_in = input_layer->template tensor<float,3>();
		auto pooled_out = output_tensor->template tensor<float,4>();
		auto region_props = proposal_regions->template tensor<float,2>();

		// Initialize output to most negative numbers possible (even if not most negative possible,
		// these will suffice)
		pooled_out.setConstant(std::numeric_limits<float>::min());
		
		for (int a = 0; a < num_rois; a++) {
			for (int b = 0; b < pooled_h; b++) {
				for(int c = 0; c < pooled_w; c++) {
					for(int d = 0; d < feat_c; d++) {
						pooled_features(a,b,c,d)=std::numeric_limits<float>::min();
					}
				}
			}
		}


		for( int n=0 ; n < num_rois ; n++ ) {
		    // Region of interest translated to feature input
			const int roi_w_feat_start = static_cast<int>(
							std::floor(region_props(n,0)/true_feat_stride + .5));
			const int roi_h_feat_start = static_cast<int>(
							std::floor(region_props(n,1)/true_feat_stride + .5));
			const int roi_w_feat_end = static_cast<int>(
							std::ceil(region_props(n,2)/true_feat_stride + .5));
			const int roi_h_feat_end = static_cast<int>(
							std::ceil(region_props(n,3)/true_feat_stride + .5));
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
						
						if (ph>1 || pw>1) {
						}
						// Incase we get a bad region
						if ( (h_f <= h_0) || (w_f <= w_0) ){
						    pooled_out(n, ph, pw, c) = 0.;
						}

						for ( int h=h_0 ; h<h_f ; h++ ) {
						    for ( int w=w_0 ; w<w_f ; w++ ) {
								if ( feat_in(h, w, c) > pooled_out(n, ph, pw, c) ){
								    pooled_out(n, ph, pw, c) = feat_in(h, w, c);
								}
							}
						}
					}
				}
			}
		}
    }
};
 
REGISTER_KERNEL_BUILDER(\
		Name("RoiPooler")\
		.Device(DEVICE_CPU)\
		,RoiPooler);
