import torch

def box_encode(boxlists, output_h, output_w, num_classes, aug=False):
    N = len(boxlists)
    outputs = []
    for boxlist in boxlists:
        output = torch.zeros(num_classes, output_h, output_w, dtype=torch.float, device=boxlist.bbox.device)
        bboxs = boxlist.bbox
        labels = boxlist.get_field("labels").long() - 1
        for bbox, label in zip(bboxs, labels):
            x0, y0, x1, y1 = bbox
            w = x1 - x0
            h = y1 - y0
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            y = torch.arange(0, output_h, dtype=torch.float, device=output.device)
            x = torch.arange(0, output_w, dtype=torch.float, device=output.device)
            y, x = torch.meshgrid(y, x)
            color = 1 - torch.max(torch.abs(x - cx) / w, torch.abs(y - cy) / h)
            color = color * (color >= 0.5).float()
            if aug:
                color = color * (torch.rand(1, dtype=color.dtype, device=color.device) * 2).clamp(max=1)
            output[label] = torch.max(output[label], color)
        outputs.append(output)
    return torch.stack(outputs)


def mask_encode(boxlists, output_h, output_w, num_classes, aug=False):
    N = len(boxlists)
    outputs = []
    for boxlist in boxlists:
        output = torch.zeros(num_classes, output_h, output_w, dtype=torch.float, device=boxlist.bbox.device)
        bboxs = boxlist.bbox
        labels = boxlist.get_field("labels") - 1
        if "masks" in boxlist.extra_fields:
            masks = boxlist.get_field("masks")
            for bbox, label, mask in zip(bboxs, labels, masks):
                x0, y0, x1, y1 = bbox
                w = x1 - x0
                h = y1 - y0
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                y = torch.arange(0, output_h, dtype=torch.float, device=output.device)
                x = torch.arange(0, output_w, dtype=torch.float, device=output.device)
                y, x = torch.meshgrid(y, x)
                color = 1 - torch.max(torch.abs(x - cx) / w, torch.abs(y - cy) / h)
                color = color * (color >= 0.5).float()
                if aug:
                    color = color * (torch.rand(1, dtype=color.dtype, device=color.device) * 2).clamp(max=1)
                mask = mask.convert(mode="mask").to(color.device)
                mask_h, mask_w = mask.size()
                color[:mask_h, :mask_w] = color[:mask_h, :mask_w] * mask.float()
                output[label] = torch.max(output[label], color)

        outputs.append(output)
    return torch.stack(outputs)
