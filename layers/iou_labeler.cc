#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

REGISTER_OP("IouLabeler")
	.Input("bbox_in : float")
	.Input("gt_in : float")
	.Attr("iou_threshold : float")
	.Output("labeled_bbox : float")

static inline void ParseAndCheckBoxSizes(OpKernelContext* context,
		const Tensor& boxes, const Tensor& gt) {
	OP_REQUIRES(context, boxes.dims() == 2,
			errors::InvalidArgument("Boxes must be 2-d",
				boxes.shape().DebugString()));
	OP_REQUIRES(context, boxes.dim_size(1) == 4,
			errors::InvalidArgument("Boxes must have 4 columns"));
	OP_REQUIRES(context, gt.dims() == 2,
			errors::InvalidArgument("Ground truth must be 2-d",
				gt.shape().DebugString()));
	OP_REQUIRES(context, gt.dim_size(1) == 4,
			errors::InvalidArgument("Ground truth must have 4 columns"));
}

static inline float ComputeIOU(typename TTypes<float, 2>::ConstTensor boxes,
		typename TTypes<float, 2>::ConstTensor gt, int i, int j) {
	const float ymin_box = std::min<float>(boxes(i, 0), boxes(i, 2));
	const float xmin_box = std::min<float>(boxes(i, 1), boxes(i, 3));
 	const float ymax_box = std::max<float>(boxes(i, 0), boxes(i, 2));
 	const float xmax_box = std::max<float>(boxes(i, 1), boxes(i, 3));
 	const float ymin_gt = std::min<float>(gt(j, 0), gt(j, 2));
 	const float xmin_gt = std::min<float>(gt(j, 1), gt(j, 3));
 	const float ymax_gt = std::max<float>(gt(j, 0), gt(j, 2));
 	const float xmax_gt = std::max<float>(gt(j, 1), gt(j, 3));

 	// Use modified formula for area calculations to match Faster R-CNN 
 	const float area_box = (ymax_box - ymin_box + 1.0) * (xmax_box - xmin_box + 1.0);
 	const float area_gt = (ymax_gt - ymin_gt + 1.0) * (xmax_gt - xmin_gt + 1.0);
  	if (area_box <= 0 || area_gt <= 0) return 0.0;
 	const float intersection_ymin = std::max<float>(ymin_box, ymin_gt);
 	const float intersection_xmin = std::max<float>(xmin_box, xmin_gt);
 	const float intersection_ymax = std::min<float>(ymax_box, ymax_gt);
 	const float intersection_xmax = std::min<float>(xmax_box, xmax_gt);

 	// Use modified formula for area calculations to match Faster R-CNN
  	const float intersection_area =
     	std::max<float>(intersection_ymax - intersection_ymin + 1.0, 0.0) *
     	std::max<float>(intersection_xmax - intersection_xmin + 1.0, 0.0);
  	return intersection_area / (area_box + area_gt - intersection_area);
}

class IouLabelerOp : public OpKernel {
	public:
		explicit IouLabelerOp(OpKernelConstruction* context)
			: OpKernel(context) { 
			OP_REQUIRES_OK(context, context->getAttr("iou_threshold", &_iou_threshold_));
		}

		void Compute(OpKernelContext* context) override {
			OP_REQUIRES(context, iou_threshold_ >= 0 && iou_threshold_ <= 1,
					errors::InvalidArgument("iou_threshold must be in interval [0,1]"));

			const Tensor& _boxes = context->input(0);
			const Tensor& _gt = context->input(1);

			ParseAndCheckBoxSizes(context, boxes, gt);
			if (!context->status().ok()) return;

			typename TTypes<float, 2>::ConstTensor boxes = _boxes.tensor<float, 2>();
			typename TTypes<float, 2>::ConstTensor gt = _gt.tensor<float, 2>();

			const int num_boxes = _boxes.dim_size(0);
			const int num_gts = _gt.dim_size(0);
			Tensor* _out = nullptr;
			TensorShape out_shape({num_boxes, 5});
			OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &_out));
			
			typename TTypes<float, 3>::ConstTensor out = _boxes.tensor<float, 3>();

			for(int i=0; i < num_boxes; ++i) {
				float max_IoU = 0.;
				int argmax_IOU = num_gts;
				for(int j=0; j < num_gts; ++j) {
					float IoU = ComputeIOU(boxes, gt, i, j);
					if (IoU >= iou_threshold_ && IoU >= max_IoU) {
						max_IoU = IoU;
						argmax_IOU = j;
					}
					out(i, 0) = boxes(i, 0);
					out(i, 1) = boxes(i, 1);
					out(i, 2) = boxes(i, 2);
					out(i, 3) = boxes(i, 3);
					// The output type is unfortunately float, so this will have to do
					out(i, 4) = static_cast<float>(argmax_IoU);
				}
			}
		}
	private:
		float iou_threshold_;
};
				
REGISTER_KERNEL_BUILDER(\
		Name("IouLabeler")\
		.Device(DEVICE_CPU)\
		,IouLabelerOp);
