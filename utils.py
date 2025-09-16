import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import MushroomClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def grad_cam(image_path, model_path='./models/mushroom_classifier.pth'):
    model = MushroomClassifier().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    hook = model.conv2.register_forward_hook(forward_hook)

    output = model(input_tensor)
    class_idx = output.argmax().item()

    model.zero_grad()
    output[0, class_idx].backward()

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    activations = activations[0][0]

    for i in range(pooled_gradients.size(0)):
        activations[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img
    output_path = f'./outputs/grad_cam_output.jpg'
    cv2.imwrite(output_path, superimposed_img)
    hook.remove()

    plt.imshow(cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return output_path
