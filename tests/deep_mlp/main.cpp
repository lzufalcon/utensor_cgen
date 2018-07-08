#include "simple_mnist.hpp"
#include "tensorIdxImporter.hpp"
#include "tensor.hpp"
#include <string>

void run_mlp()
{
    TensorIdxImporter t_import;
    Tensor *input_x = t_import.float_import("tmp.idx");
    Context ctx;

    get_simple_mnist_ctx(ctx, input_x);
    S_TENSOR pred_tensor = ctx.get("y_pred:0");
    ctx.eval();

    int pred_label = *(pred_tensor->read<int>(0, 0));
    
    printf("Predicted label: %d", pred_label);
}

int main(void)
{
    init_env();
    printf("Simple MNIST end-to-end uTensor cli example on pc\n");
    run_mlp();

    return 0;
}
