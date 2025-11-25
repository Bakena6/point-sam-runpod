"""
  Point-SAM RunPod Serverless Handler
  ===================================
  3D Point Cloud Segmentation via API
  """

  import runpod
  import torch
  import numpy as np
  import base64
  import io
  import sys
  import os

  sys.path.insert(0, '/app/point-sam')

  # Global model (loaded once)
  MODEL = None
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


  def load_model():
      """Load Point-SAM model"""
      global MODEL
      if MODEL is not None:
          return MODEL

      from model import PointSAM

      MODEL = PointSAM(
          encoder="vit_l",
          checkpoint="/app/weights/model.safetensors"
      )
      MODEL.to(DEVICE)
      MODEL.eval()

      print(f"Point-SAM loaded on {DEVICE}")
      return MODEL


  def decode_point_cloud(data_b64: str) -> np.ndarray:
      """Decode base64 point cloud (N x 3 or N x 6 with colors)"""
      data = base64.b64decode(data_b64)
      points = np.frombuffer(data, dtype=np.float32)

      # Reshape: try N x 6 (with colors), else N x 3
      if len(points) % 6 == 0:
          points = points.reshape(-1, 6)
      else:
          points = points.reshape(-1, 3)

      return points


  def encode_mask(mask: np.ndarray) -> str:
      """Encode segmentation mask to base64"""
      return base64.b64encode(mask.astype(np.uint8).tobytes()).decode()


  def handler(job):
      """
      RunPod handler for Point-SAM
      
      Input:
          - point_cloud_b64: base64 encoded point cloud (float32, N x 3)
          - prompt_points: list of [x, y, z] prompt points
          - prompt_labels: list of labels (1=foreground, 0=background)
      
      Output:
          - masks: list of base64 encoded masks
          - scores: confidence scores
      """
      job_input = job["input"]

      try:
          model = load_model()

          # Get inputs
          point_cloud_b64 = job_input.get("point_cloud_b64")
          prompt_points = job_input.get("prompt_points", [])
          prompt_labels = job_input.get("prompt_labels", [])

          if not point_cloud_b64:
              return {"error": "point_cloud_b64 is required"}

          # Decode point cloud
          points = decode_point_cloud(point_cloud_b64)
          points_tensor = torch.from_numpy(points[:, :3]).float().unsqueeze(0).to(DEVICE)

          # Prepare prompts
          if prompt_points:
              prompt_tensor = torch.tensor(prompt_points).float().unsqueeze(0).to(DEVICE)
              labels_tensor = torch.tensor(prompt_labels).long().unsqueeze(0).to(DEVICE)
          else:
              # Auto mode - no prompts
              prompt_tensor = None
              labels_tensor = None

          # Run inference
          with torch.no_grad():
              if prompt_tensor is not None:
                  masks, scores = model(
                      points_tensor,
                      prompt_points=prompt_tensor,
                      prompt_labels=labels_tensor
                  )
              else:
                  masks, scores = model.segment_auto(points_tensor)

          # Convert results
          masks_np = masks.cpu().numpy()[0]  # (num_masks, N)
          scores_np = scores.cpu().numpy()[0]  # (num_masks,)

          results = []
          for i in range(len(scores_np)):
              results.append({
                  "mask_b64": encode_mask(masks_np[i]),
                  "score": float(scores_np[i]),
                  "num_points": int(masks_np[i].sum())
              })

          # Sort by score
          results.sort(key=lambda x: x["score"], reverse=True)

          return {
              "success": True,
              "num_input_points": len(points),
              "num_masks": len(results),
              "masks": results[:5]  # Top 5 masks
          }

      except Exception as e:
          return {
              "success": False,
              "error": str(e)
          }


  # Start RunPod serverless
  runpod.serverless.start({"handler": handler})
