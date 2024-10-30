import torch
from einops import rearrange
import torch.nn.functional as F
import torchvision.transforms as transforms



def aggregate_attention_maps(attention_store):
    attention_dict = {}
    for attention_map in attention_store:
        b,h,w,f = attention_map.shape
        if h not in attention_dict.keys():
            attention_dict[h] = [attention_map]
        else:
            attention_dict[h].append(attention_map)
    return attention_dict


def estimate_optical_flow_raft(raft_model, video_batch):

    b, c, f, h, w = video_batch.shape

    if h < 128:
        # Resize the video to at least 128x128
        transform = transforms.Resize((128, 128))
        video_batch = rearrange(video_batch, 'b c f h w -> (b f) c h w')
        video_batch = torch.stack([transform(frame) for frame in video_batch], dim=0)
        video_batch = rearrange(video_batch, '(b f) c h w -> b c f h w', b=b)
        h, w = 128, 128
        

    # Prepare to store optical flow
    optical_flows = []

    for i in range(b):
        batch_optical_flow = []
        for t in range(f - 1):
            # Get consecutive frames
            frame1 = video_batch[i, :, t].unsqueeze(0)
            frame2 = video_batch[i, :, t + 1].unsqueeze(0)

            if c == 1:
                frame1 = frame1.repeat(1, 3, 1, 1)
                frame2 = frame2.repeat(1, 3, 1, 1)
            
            # Predict the optical flow using RAFT
            # with torch.no_grad():
            flow = raft_model(frame1, frame2)[-1]  # Get the final flow prediction
            
            batch_optical_flow.append(flow.squeeze(0))  # Remove batch dimension
        
        # Stack the flows for the current batch
        batch_optical_flow = torch.stack(batch_optical_flow, dim=0)
        optical_flows.append(batch_optical_flow)
    
    # Stack the flows for all batches
    optical_flows = torch.stack(optical_flows, dim=0)
    return optical_flows


def eot_loss_with_video(raft_model,x0,attention_store):
    

    attention_dict = aggregate_attention_maps(attention_store)

    x0 = rearrange(x0,"b f c h w -> b c f h w")
    x0_optical_flow = estimate_optical_flow_raft(raft_model,x0) # b,f-1,1,2,h,w
    
    all_cos_similarity = []
    for h,attention_maps in attention_dict.items():
        attention_maps = torch.stack(attention_maps,dim=0).mean(dim=0)
        attention_maps = rearrange(attention_maps,'b h w f -> b 1 f h w')
    
        attn_optical_flow = estimate_optical_flow_raft(raft_model,attention_maps) # b,f-1,2,h,w
        
        _,_,_,h,w = attn_optical_flow.shape

        transform = transforms.Resize((h,w))
        resized_x0_optical_flow = torch.stack([transform(flow) for flow in x0_optical_flow],dim=0)
        
        attn_optical_flow = rearrange(attn_optical_flow,'b f i h w -> b (f i h w)')
        resized_x0_optical_flow = rearrange(resized_x0_optical_flow,'b f i h w -> b (f i h w)')
        
        cos_similarity = F.cosine_similarity(attn_optical_flow, resized_x0_optical_flow, dim=1)
        all_cos_similarity.append(cos_similarity)
        
    avg_cos_similarity = torch.stack(all_cos_similarity,dim=0).mean(dim=0)
    
    return -avg_cos_similarity
    