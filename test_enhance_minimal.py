import torch
from PIL import Image
from torchvision import transforms

from models.networks import ParseNet, apply_norm
from models.psfrnet import PSFRGenerator
from utils import utils

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


if __name__ == '__main__':

    netP = ParseNet(in_size=512, out_size=512, relu_type="LeakyReLU", ch_range=[32, 256])
    netP.eval()
    netP.load_state_dict(torch.load("./pretrain_models/parse_multi_iter_90000.pth"))
    netP.to("cpu")

    netG = PSFRGenerator(input_nc=3, output_nc=3, in_size=512, out_size=512, relu_type="LeakyReLU", parse_ch=19, norm_type="spade")
    apply_norm(netG, "spectral_norm")
    netG.eval()
    netG.load_state_dict(torch.load("./pretrain_models/psfrgan_epoch15_net_G.pth"), strict=False)
    netG.to("cpu")


    img_input = Image.open("test_dir/lowres512.jpg").convert('RGB')
    #img_input = img_input.resize((512, 512), Image.Resampling.LANCZOS)
    img_input_tensor = image_transform(img_input).unsqueeze(0)

    with torch.no_grad():
        parse_map, _ = netP(img_input_tensor)
        parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
        output_SR = netG(img_input_tensor, parse_map_sm)


    output_sr_img = utils.batch_tensor_to_img(output_SR)
    save_img = Image.fromarray(output_sr_img[0])
    print("enhancement finished")
    save_img.save("./test_dir/hires512.jpg")
