import argparse
import os
from ultralytics import YOLO

def export_model(weights_path, img_size=(1280, 1280)):
    """
    Exports a YOLO model to ONNX with optimizations for TensorRT.
    """
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        return

    print(f"--- Loading model from {weights_path} ---")
    
    try:
        # The YOLO class automatically handles the 'weights_only' security 
        # issues and 'DetectionModel' globals for you.
        model = YOLO(weights_path)

        print(f"--- Exporting to ONNX (imgsz={img_size}) ---")
        
        # We use opset=12 for maximum compatibility with Jetson JetPack 4.x/5.x
        # 'simplify=True' removes redundant nodes that can confuse TensorRT
        success = model.export(
            format='onnx',
            imgsz=img_size,
            opset=12,
            simplify=True,
            dynamic=False  # Static shapes are generally faster on Jetson/TensorRT
        )
        
        if success:
            onnx_path = weights_path.replace('.pt', '.onnx')
            print(f"--- Export successful: {onnx_path} ---")
            print(f"\nNext, run this on your Jetson device:")
            print(f"trtexec --onnx={onnx_path} --saveEngine=model.trt --fp16 --workspace=2048")
            
    except Exception as e:
        print(f"An error occurred during export: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX for Edge Deployment")
    parser.add_argument('--model', type=str, required=True, help='Path to .pt file')
    parser.add_argument('--img-size', nargs='+', type=int, default=[1280, 1280], help='Image size [h, w]')
    
    args = parser.parse_args()
    
    # Format img_size correctly
    img_sz = tuple(args.img_size) if len(args.img_size) == 2 else (args.img_size[0], args.img_size[0])
    
    export_model(args.model, img_sz)