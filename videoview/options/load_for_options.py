


from ..data import create_dataloader, create_dataset
from .options import parse_yml
from ..imports import *
import importlib

def get_dataloader_from_yml(yml_file_path):
    if yml_file_path is None:
        raise Exception("need yml file")
    ops = parse_yml(yml_file_path)
    train_dataset = create_dataset(
        ops["datasets"]["train"]["mode"],
        ops["datasets"]["train"]["dataroot_LQ"],
        ops["datasets"]["train"]["dataroot_GT"],
        ops["datasets"]["train"]["GT_size"],
        ops["scale"],
        ops["datasets"]["train"]["sequence_length"],
    )
    val_dataset = create_dataset(
        ops["datasets"]["val"]["mode"],
        ops["datasets"]["val"]["dataroot_LQ"],
        ops["datasets"]["val"]["dataroot_GT"],
        ops["datasets"]["val"]["GT_size"],
        ops["scale"],
        ops["datasets"]["val"]["sequence_length"]
    )
    train_loader = create_dataloader(
        train_dataset,
        "train",
        ops["datasets"]["train"]["n_workers"],
        ops["datasets"]["train"]["batch_size"],
        ops["datasets"]["train"]["use_shuffle"],
    )
    val_loader = create_dataloader(val_dataset, "validation", **ops["datasets"]["val"])
    return {
        "train_dataset": train_dataset,
        "validation_dataset": val_dataset,
        "train_dataloader": train_loader,
        "validation_dataloader": val_loader,
    }

def load_partial(generator, path,strict=False, key=None):
    model_dict = generator.state_dict()
    if key is not None:
        pretrained_dict = torch.load(path, map_location=device)[key]
    else:
        pretrained_dict = torch.load(path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    generator.load_state_dict(model_dict, strict=strict)
    return generator

def get_generator_from_yml(yml_file_path, pretrain_path=None, key=None, strict=True):
    if yml_file_path is None:
        raise Exception("need yml file")
    opt = parse_yml(yml_file_path)

    in_c = opt["structure"]["network_G"]["in_nc"]
    out_c = opt["structure"]["network_G"]["out_nc"]
    nf = opt["structure"]["network_G"]["nf"]
    num_modules = opt["structure"]["network_G"]["num_modules"]
    scale = opt["scale"]
    model_name = opt["structure"]["network_G"]["which_model_G"]
    
    if scale in [2, 4, 8, 16, 3, 6]:
        pass
    else:
        scale = 4
    net = importlib.import_module(
        "videoview.networks.{}".format(model_name)
    ).VideoEnhancer
    model = net(
        in_nc=in_c,
        nf=nf,
        nb=num_modules,
        out_nc=out_c,
        scale=scale,
    )
    model = model.to(device)

    if pretrain_path is False:
        return model
    if pretrain_path is not None:
        model = load_partial(model.to(device), pretrain_path, strict, key)
        print(f"generator is loaded from {pretrain_path}")
        model.to(device)
    else:
        if opt["pretraining_settings"]["network_G"]["want_load"] is True:
            pretrain_path = opt["pretraining_settings"]["network_G"][
                "pretrained_model_path"
            ]
            strict = opt["pretraining_settings"]["network_G"]["pretrained_model_path"]
            key = opt["pretraining_settings"]["network_G"]["key"]
            model = load_partial(model.to(device), pretrain_path, strict, key)
            print(f"generator is loaded from {pretrain_path}")
            model.to(device)
    fnet_path = opt["pretraining_settings"]["network_G"].get("fnet_model_path",None)
    if fnet_path is not None:
        print("loading optical flow weights",fnet_path)
        model.fnet.load_state_dict(torch.load(fnet_path,map_location=device))
    return model

# def load_pipeline_from_yml(yml_file_path):
#     # load dataloader
#     print("loading dataloader...")
#     loaders = get_dataloader_from_yml(yml_file_path)
#     train_loader = loaders["train_dataloader"]
#     val_loader = loaders["validation_dataloader"]

#     # load model
#     print("loading models...")
#     generator = get_generator_from_yml(yml_file_path)
#     # load discriminator
#     discriminator = get_discriminator_from_yml(yml_file_path)
#     # load trainer
#     print("constructing trainers .....")
#     trainer = get_trainer_from_yml(
#         yml_file_path=yml_file_path,
#         model_G=generator,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         model_D=discriminator,
#     )

#     return {
#         "loaders": loaders,
#         "generator": generator,
#         "critic": discriminator,
#         "trainer": trainer,
#     }