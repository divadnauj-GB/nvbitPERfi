import os, sys
import numpy as np
import copy
import torch
import torch.nn as nn
import h5py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

DEBUG=1

torch.backends.cudnn.enabled = True
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False
# print(self.Input_dataset[0:128].shape)

class extract_embeddings_nvbit:
    def __init__(self, model, lyr_type=[nn.Conv2d], lyr_num=0, batch_size=1) -> None:
        self.DNN = model
        self.batch_size = batch_size
        self.layer_id = 0
        self.extracted_layer = None
        self.layer_types = lyr_type
        self.layer_number = lyr_num
        self.Input_dataset = None
        self.Golden_dataset = None
        self.Results_dataset = None
        self.DNN_targets = []
        self.DNN_outputs = []

        self.layer_embedding_list_input = {}
        self.layer_embedding_list_output = {}
        self.handles, self.layer_model = self._traverse_model_set_hooks_neurons(
            self.DNN
        )
        print(self.layer_id, self.layer_number)

    def _get_layer_embeddings(self, name):
        def hook(_, input, output):
            if name not in self.layer_embedding_list_input:
                self.layer_embedding_list_input[name] = []
                self.layer_embedding_list_output[name] = []
            self.layer_embedding_list_input[name].append(copy.deepcopy(input[0]))
            self.layer_embedding_list_output[name].append(copy.deepcopy(output.detach()))

        return hook

    def _traverse_model_set_hooks_neurons(self, model):
        handles = []
        for layer in model.children():
            # leaf node
            if list(layer.children()) == []:
                if "all" in self.layer_types:
                    if self.layer_number == self.layer_id:
                        self.extracted_layer = copy.deepcopy(layer)
                        hook = self._get_layer_embeddings(f"layer_id_{self.layer_id}")
                        handles.append(layer.register_forward_hook(hook))
                    self.layer_id += 1
                else:
                    for i in self.layer_types:
                        if isinstance(layer, i):
                            if self.layer_number == self.layer_id:
                                self.extracted_layer = copy.deepcopy(layer)
                                hook = self._get_layer_embeddings(
                                    f"layer_id_{self.layer_id}"
                                )
                                handles.append(layer.register_forward_hook(hook))
                            self.layer_id += 1
            # unpack node
            else:
                subHandles = self._traverse_model_set_hooks_neurons(layer)
                for i in subHandles:
                    handles.append(i)
        return handles, self.extracted_layer

    def DNN_inference(self, input, targets):
        Outputs = self.DNN(input)
        self.DNN_targets.append(targets)
        self.DNN_outputs.append(Outputs)
        return Outputs

    def _save_target_layer(
        self, state, checkpoint="./checkpoint", filename="checkpoint.pth.tar"
    ):
        os.system(f"mkdir -p {checkpoint}")
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)

    def extract_embeddings_target_layer(self):
        for layer_id in self.layer_embedding_list_input:
            print(len(self.layer_embedding_list_input[layer_id]))
            print(len(self.layer_embedding_list_output[layer_id]))
            print(self.layer_model)

            print((self.layer_embedding_list_input[layer_id][0].shape))
            print((self.layer_embedding_list_output[layer_id][0].shape))

            current_path = os.path.dirname(__file__)
            embeddings_input = (
                torch.cat(self.layer_embedding_list_input[layer_id]).cpu().numpy()
            )
            embeddings_output = (
                torch.cat(self.layer_embedding_list_output[layer_id]).cpu().numpy()
            )
            DNN_targets = torch.cat(self.DNN_targets).cpu().numpy()

            DNN_Outputs = torch.cat(self.DNN_outputs).cpu().numpy()

            log_path_file = os.path.join(
                current_path, f"embeddings.h5"
            )

            with h5py.File(log_path_file, "w") as hf:
                hf.create_dataset(
                    "layer_input", data=embeddings_input, compression="gzip"
                )
                hf.create_dataset(
                    "layer_output", data=embeddings_output, compression="gzip"
                )
                hf.create_dataset("DNN_targets", data=DNN_targets, compression="gzip")
                hf.create_dataset("DNN_outputs", data=DNN_Outputs, compression="gzip")
                hf.create_dataset(
                    "sample_id",
                    data=range(len(self.layer_embedding_list_input[layer_id])),
                    compression="gzip",
                )
                # hf.create_dataset('batch_size', data=self.batch_size, compression="gzip")
            
            layer_inputs_path_file = os.path.join(
                current_path, f"inputs_layer.h5"
            )

            with h5py.File(layer_inputs_path_file, "w") as hf:
                hf.create_dataset(
                    "layer_input", data=embeddings_input, compression="gzip"
                )
            
            layer_inputs_path_file = os.path.join(
                current_path, f"Golden_Output_layer.h5"
            )

            with h5py.File(layer_inputs_path_file, "w") as hf:
                hf.create_dataset(
                    "layer_output", data=embeddings_output, compression="gzip"
                )

            self._save_target_layer(self.layer_model, filename="target_layer.pth.tar")


class load_embeddings:
    def __init__(self, layer_number, batch_size=1) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.layer_results = []
        self.layer_number = layer_number
        current_path = os.path.dirname(__file__)
        model_file = os.path.join(current_path, "checkpoint", "target_layer.pth.tar")
        # dataset_file = os.path.join(
        #     current_path,
        #     f"embeddings_batch_size_{self.batch_size}_layer_id_{self.layer_number}.h5",
        # )
        dataset_file = os.path.join(
            current_path,
            f"inputs_layer.h5",
        )
        #self.layer_model = torch.load(model_file, map_location=torch.device("cpu"))
        self.layer_model = torch.load(model_file)
        self.layer_model = self.layer_model.to(self.device)
        self.layer_model.eval()

        if (DEBUG): print(self.layer_model)

        with h5py.File(dataset_file, "r") as hf:
            self.Input_dataset = np.array(hf["layer_input"])
            #self.Output_dataset = np.array(hf["layer_output"])
            # self.batch_size=np.array(hf['batch_size'])

        if (DEBUG): print(len(self.Input_dataset))
        #print(len(self.Output_dataset))

        if (DEBUG): print((self.Input_dataset.shape))
        #print((self.Output_dataset.shape))
        # print(self.batch_size)

        self.onnx_model_name = "Layer_pytorch.onnx"
        self.TRT_model_name = "Layer_pytorch.rtr"
        self.TRT_output_shape = None

    def layer_inference(self):
        max_batches = float(float(len(self.Input_dataset)) / float(self.batch_size))
        with torch.no_grad():
            for batch in range(0, int(np.ceil(max_batches))):
                img = self.Input_dataset[
                    batch * self.batch_size : batch * self.batch_size + self.batch_size
                ]
                img_tensor = torch.from_numpy(img)
                img_tensor = img_tensor.to(self.device)
                # targets = self.Output_dataset[
                #     batch * self.batch_size : batch * self.batch_size + self.batch_size
                # ]

                out = self.layer_model(img_tensor)
                if batch == 0: self.TRT_output_shape = out.shape
                """
                Golden_output = (
                    torch.from_numpy(
                        self.Output_dataset[
                            batch * self.batch_size : batch * self.batch_size
                            + self.batch_size
                        ]
                    )
                ).to(self.device)
                """
                # if not torch.equal(out, Golden_output):
                #    print("Not getting the expected result!")
                # np_out = out.cpu().detach().numpy()

                self.layer_results.append(out.cpu().detach())

                #print(img_tensor.shape)
                #print(out.shape)
                # print(Golden_output.shape)
                # print(torch.eq(out,Golden_output))
                # print(targets-np_out)
                # break
        embeddings_outputs = torch.cat(self.layer_results).numpy()

        current_path = os.path.dirname(__file__)
        log_path_file = os.path.join(
            current_path,
            f"Output_layer.h5",
        )

        with h5py.File(log_path_file, "w") as hf:
            hf.create_dataset(
                "layer_output", data=embeddings_outputs, compression="gzip"
            )

    def TRT_create_onnx_model(self):
        max_batches = float(float(len(self.Input_dataset)) / float(self.batch_size))
        batch = 0
        img = self.Input_dataset[
            batch * self.batch_size : batch * self.batch_size + self.batch_size
        ]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(self.device)
        torch.onnx.export(self.layer_model, img_tensor, self.onnx_model_name, verbose=False)
        USE_FP16 = False
        target_dtype = np.float16 if USE_FP16 else np.float32
        if USE_FP16:
            cmd=f"/usr/src/tensorrt/bin/trtexec --onnx={self.onnx_model_name} --saveEngine={self.TRT_model_name} --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 "
        else:
            cmd=f"/usr/src/tensorrt/bin/trtexec --onnx={self.onnx_model_name} --saveEngine={self.TRT_model_name} --explicitBatch "
        
        msg=os.system(cmd)
        print(cmd)


    def TRT_layer_inference(self):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        self.target_dtype = np.float32
        current_path = os.path.dirname(__file__)
        
        with open(os.path.join(current_path, self.TRT_model_name), "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
            engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()
            output = np.empty(self.TRT_output_shape, dtype = self.target_dtype) 
            if DEBUG: print(self.TRT_output_shape)
            if DEBUG: print(output.shape)
            batch = 0
            sample_image = self.Input_dataset[
                        batch * self.batch_size : batch * self.batch_size + self.batch_size
                    ]
            d_input = cuda.mem_alloc(1 * sample_image.nbytes)
            d_output = cuda.mem_alloc(1 * output.nbytes)
            bindings = [int(d_input), int(d_output)]
            stream = cuda.Stream()

            max_batches = float(float(len(self.Input_dataset)) / float(self.batch_size))

            with torch.no_grad():
                for batch in range(0, int(np.ceil(max_batches))):
                    img = self.Input_dataset[
                        batch * self.batch_size : batch * self.batch_size + self.batch_size
                    ]
                    cuda.memcpy_htod_async(d_input, img, stream)
                    # execute model
                    context.execute_async_v2(bindings, stream.handle, None)
                    # transfer predictions back
                    cuda.memcpy_dtoh_async(output, d_output, stream)
                    # syncronize threads

                    """
                    Golden_output = (
                        torch.from_numpy(
                            self.Output_dataset[
                                batch * self.batch_size : batch * self.batch_size
                                + self.batch_size
                            ]
                        )
                    ).to(self.device)
                    """
                    # if not torch.equal(out, Golden_output):
                    #    print("Not getting the expected result!")
                    # np_out = out.cpu().detach().numpy()
                    self.layer_results.append(torch.from_numpy(output))

                    #print(img_tensor.shape)
                    #print(out.shape)
                    # print(Golden_output.shape)
                    # print(torch.eq(out,Golden_output))
                    # print(targets-np_out)
                    # break
            embeddings_outputs = torch.cat(self.layer_results).numpy()

            if DEBUG: print(embeddings_outputs.shape)

            current_path = os.path.dirname(__file__)
            log_path_file = os.path.join(
                current_path,
                f"Output_layer.h5",
            )

            with h5py.File(log_path_file, "w") as hf:
                hf.create_dataset(
                    "layer_output", data=embeddings_outputs, compression="gzip"
                )


class TRT_load_embeddings:
    def __init__(self, layer_number, batch_size=1, layer_output_shape=(1,)) -> None:
        self.target_dtype = np.float32
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.layer_results = []
        self.layer_number = layer_number
        self.onnx_model_name = "Layer_pytorch.onnx"
        self.TRT_model_name = "Layer_pytorch.rtr"
        self.TRT_output_shape = layer_output_shape

        current_path = os.path.dirname(__file__)
        # dataset_file = os.path.join(
        #     current_path,
        #     f"embeddings_batch_size_{self.batch_size}_layer_id_{self.layer_number}.h5",
        # )
        dataset_file = os.path.join(
            current_path,
            f"inputs_layer.h5",
        )
        with h5py.File(dataset_file, "r") as hf:
            self.Input_dataset = np.array(hf["layer_input"])
            #self.Output_dataset = np.array(hf["layer_output"])
            # self.batch_size=np.array(hf['batch_size'])
            
        #self.layer_model = torch.load(model_file, map_location=torch.device("cpu"))
        with open(os.path.join(current_path, self.TRT_model_name), "rb") as f:
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.output = np.empty(self.TRT_output_shape, dtype = self.target_dtype) 
            if DEBUG: print(self.TRT_output_shape)
            if DEBUG: print(self.output.shape)
            batch = 0
            sample_image = self.Input_dataset[
                        batch * self.batch_size : batch * self.batch_size + self.batch_size
                    ]
            self.d_input = cuda.mem_alloc(1 * sample_image.nbytes)
            self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
            self.bindings = [int(self.d_input), int(self.d_output)]
            self.stream = cuda.Stream()
            # warming up
            # output=self.__TRT_forward_function(sample_image)
            # end warming up

        if (DEBUG): print(len(self.Input_dataset))
        #print(len(self.Output_dataset))

        if (DEBUG): print((self.Input_dataset.shape))
        #print((self.Output_dataset.shape))
        # print(self.batch_size)

    def __TRT_forward_function(self,input):
        cuda.memcpy_htod_async(self.d_input, input, self.stream)
        # execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # syncronize threads
        self.stream.synchronize()
        out = self.output
        return (out)

    def TRT_layer_inference(self):
        max_batches = float(float(len(self.Input_dataset)) / float(self.batch_size))
        with torch.no_grad():
            for batch in range(0, int(np.ceil(max_batches))):
                img = self.Input_dataset[
                    batch * self.batch_size : batch * self.batch_size + self.batch_size
                ]
                output=self.__TRT_forward_function(img)
                self.layer_results.append(torch.from_numpy(output))

        embeddings_outputs = torch.cat(self.layer_results).numpy()

        if DEBUG: print(embeddings_outputs.shape)

        current_path = os.path.dirname(__file__)
        log_path_file = os.path.join(
            current_path,
            f"Output_layer.h5",
        )

        with h5py.File(log_path_file, "w") as hf:
            hf.create_dataset(
                "layer_output", data=embeddings_outputs, compression="gzip"
            )