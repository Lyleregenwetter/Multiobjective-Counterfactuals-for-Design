# import requests
# import pandas as pd
# import cairosvg
# import time
# import os
# import scipy
# from PIL import Image
# import numpy as np
# import subprocess
# import platform
# import glob
# import multiprocessing as mp
# import math
# from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
# from tqdm import trange,tqdm
# import torch
# import sys
#
# import dataFrameTools
# import processGen
# import signal
# from torchvision import transforms
# from torchvision.transforms import Compose
#
# class BikeCAD():
#     def __init__(self):
#         if platform.system() == "Windows":
#             self.expected_success = b'Done!\r\n'
#         else:
#             self.expected_success = b'Done!\n'
#
#         self.instance = self.start_bike_cad_Instance()
#
#     def start_bike_cad_Instance(self):
#         p = subprocess.Popen('java -jar console_BikeCAD_final.jar'.split(' '), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
#         p.stdout.read(14)
#         return p
#
#     def export_svgs(self, folder):
#         p = self.instance
#         p.stdin.write(bytes("svg<>" + folder + "\n",'UTF-8'));
#         p.stdin.flush();
#
#         stdout = ""
#         while stdout!=self.expected_success:
#             stdout = p.stdout.readline()
#
#     def export_pngs(self, folder):
#         p = self.instance
#         p.stdin.write(bytes("png<>" + folder + "\n",'UTF-8'));
#         p.stdin.flush();
#
#         stdout = ""
#         while stdout!=self.expected_success:
#             stdout = p.stdout.readline()
#
#     def export_svg_from_list(self,files):
#         p = self.instance
#         p.stdin.write(bytes("svglist<>" + "<>".join(files) + "\n",'UTF-8'));
#         p.stdin.flush();
#
#         stdout = ""
#         while stdout!=self.expected_success:
#             stdout = p.stdout.readline()
#
#     def export_png_from_list(self,files):
#         p = self.instance
#         p.stdin.write(bytes("pnglist<>" + "<>".join(files) + "\n",'UTF-8'));
#         p.stdin.flush();
#
#         stdout = ""
#         while stdout!=self.expected_success:
#             stdout = p.stdout.readline()
#
#     def kill(self):
#         self.instance.kill()
#
# # multi-processor function
# def run_imap_multiprocessing(func, argument_list, show_prog = True):
#     pool = mp.Pool(processes=mp.cpu_count())
#
#     if show_prog:
#         result_list_tqdm = []
#         for result in tqdm(pool.imap(func=func, iterable=argument_list), total=1,position=0, leave=True):
#             result_list_tqdm.append(result)
#     else:
#         result_list_tqdm = []
#         for result in pool.imap(func=func, iterable=argument_list):
#             result_list_tqdm.append(result)
#
#     return result_list_tqdm
#
# def init_mp(reserve_threads=2):
#     thread_count = mp.cpu_count()
#     print(f"CPU Thread Count: {thread_count}; reserving 2")
#     global bcad_instances
#     bcad_instances = []
#     for i in trange(thread_count-2):
#         bcad_instances.append(BikeCAD())
#     return thread_count-2
#
# class TimeoutException(Exception):   # Custom exception class
#
#     pass
#
# def timeout_handler(signum, frame):   # Custom signal handler
#     raise TimeoutException
#
# signal.signal(signal.SIGALRM, timeout_handler)
#
# def svg_auxilary_function(inputs):
#     i,flist = inputs
#     if len(flist)>0:
#         bcad_instances[i].export_svg_from_list(flist)
#
# def svg_auxilary_function_sig(inputs):
#     i,flist = inputs
#     if len(flist)>0:
#         signal.alarm(20+2*len(flist))
#         try:
#             bcad_instances[i].export_svg_from_list(flist)
#             print(f"Success in instance {i}!")
#             return True
#         except:
#             return False
#     else:
#         return True
#
#
# def png_auxilary_function(inputs):
#     i,flist = inputs
#     if len(flist)>0:
#         bcad_instances[i].export_png_from_list(flist)
#
# def png_convert_auxiliary_function(inputs):
#     file, num_views = inputs
#     if file:
#         targetfile = file.replace(".svg", ".png")
#         if os.path.isfile(file):
#             try:
#                 res = cairosvg.svg2png(url=file, write_to=targetfile)
#             except:
#                 return None
#             img = Image.open(targetfile).convert("RGB")
#             width,height = img.size
#             color = get_main_color(img)
#             result = Image.new(img.mode, (1070, 1070), color)
# #             print("check")
#             result.paste(img, (0,(width-height)//2,width, height+(width-height)//2))
# #             print("check2")
#             # views = get_augmented_views(result, num_views)
#             tt = transforms.ToTensor()
#             return (tt(result), color)
#
#     return None
#
# def get_image(bikes, dataset, thread_count, num_views=10, timeout=10):
#
#     num = len(bikes.index)
#     bikes.index=range(num)
#     for f in os.listdir('BCAD_gen'):
#         os.remove(os.path.join('BCAD_gen', f))
#
# #     signal.alarm(5+len(bikes)/5)
# #     try:
# #     except:
# #         pass
#     processGen.processGen(bikes, dataset=dataset, from_OH = True, check=True, sourcepath="PlainRoadbikestandardized.txt", targetpath = "BCAD_gen/")
#
#
#     files = glob.glob('BCAD_gen/*.bcad')
#     print("Number Files Being Handled: %i"%len(files))
#     batches = []
#
#     batch_size = math.ceil(len(files)/thread_count)
#
#     for i in range(thread_count):
#         batches.append([i,files[i*batch_size:(i+1)*batch_size]])
#     status = run_imap_multiprocessing(svg_auxilary_function_sig, batches, False)
#     for i in range(len(status)):
#         if not status[i]:
#             print(f"Timeout in instance {i}! Restarting instance. ")
#             pid = bcad_instances[i].instance.pid
#             os.system(f"kill {pid}")
#             bcad_instances[i] = BikeCAD()
#     files=[]
#     for i in range(num):
#         files.append([f'BCAD_gen/{i}.svg', num_views])
#     images_and_bgcol = run_imap_multiprocessing(png_convert_auxiliary_function, files, False)
#     print("finished svg conversion")
#     return images_and_bgcol
#
#
#
#
# def get_clip_embedding_views(bikes, dataset, processor, model, thread_count, timeout=10, num_views=0, batchsize=200, return_images=False):
#     start_time = time.time()
#     bikes=bikes.copy()
#     images_and_bgcol = get_image(bikes, dataset, thread_count, timeout=timeout, num_views=num_views)
#     print(f" Images loaded after {time.time() - start_time} seconds")
#
#     valid_image_tensors=[]
#     valid_idxs=[]
#     background_colors = []
#     for i in range(len(images_and_bgcol)):
#         if images_and_bgcol[i]:
#             valid_idxs.append(i)
#             valid_image_tensors.append(images_and_bgcol[i][0])
#             background_colors.append(images_and_bgcol[i][1])
#     if valid_image_tensors ==[]:
#         print("No valid designs queried!")
#         views_processed = None
#         views_emb = None
#     else:
#         all_imgs = torch.stack(valid_image_tensors)
#         views_processed = []
#         views_emb = []
#         views = []
#         for i in range(num_views+1):
#             for j in range(int(np.ceil(len(all_imgs)/batchsize))):
#                 torch.cuda.empty_cache()
#                 try:
#                     batch = all_imgs[batchsize*j:batchsize*(j+1)]
#                 except:
#                     batch = all_imgs[batchsize*j:]
#                 if i!=0:
#                     print(f"Generating view {i} batch {j}")
#                     view = get_augmented_views_gpu(batch)
#                 else:
#                     view = batch
#                 view = [view[i] for i in range(view.size()[0])]
#                 if return_images:
#                     views = views = view
#                 print(f" View {i} batch {j} generated after {time.time() - start_time} seconds")
#                 view_processed, view_emb = eval_embedding(processor, model, view)
#                 print(f" View {i} batch {j} image features calculated after {time.time() - start_time} seconds")
#                 view_processed = view_processed.cpu()
#                 view_emb = view_emb.cpu()
#                 views_processed.append(view_processed)
#                 views_emb.append(view_emb)
#         views_emb = torch.concat(views_emb)
#         views_processed = torch.concat(views_processed)
#     if return_images:
#         return valid_idxs, views_emb, views_processed, views
#     else:
#         return valid_idxs, views_emb
#
#
# def eval_embedding(processor, model, all_views):
#     with torch.no_grad():
#             device="cuda"
#             views_processed = processor(text=None,images=all_views,return_tensors='pt')['pixel_values'].to(device)
#             views_emb = model.get_image_features(views_processed)
#     return views_processed, views_emb
#
# # def eval_embedding_batch(processor, model, all_views, batchsize):
# #     with torch.no_grad():
# #         device="cuda"
# #         views_emb = []
# #         views_processed = []
# #         for i in range(len(all_views)):
# #             try:
# #                 batch = images[batchsize*i:batchsize*(i+1)]
# #             except:
# #                 batch = images[batchsize*i:]
# #             views_processed_batch = processor(text=None,images=all_views,return_tensors='pt')['pixel_values'].to(device)
# #             views_processed.append(views_processed_batch)
# #             views_emb.append(model.get_image_features(views_processed_batch))
# #     views_emb = torch.concat(views_emb)
# #     views_processed = torch.concat(views_processed)
# #     return views_emb, views_processed
#
# def get_mean_embedding(bikes, dataset, processor, model, thread_count, timeout=10, num_views=0):
#     valid_idxs, views_emb = get_clip_embedding_views(bikes, dataset, processor, model, thread_count, timeout=timeout, num_views=num_views, return_images=False)
#     if valid_idxs == []:
#         return valid_idxs, None
#     views_reshaped = torch.reshape(views_emb, (num_views+1, len(valid_idxs), 512))
#     meanvals = torch.mean(views_reshaped, axis=0)
#     all_means = [None]*len(bikes.index)
#     return valid_idxs, meanvals
#
#
#
#
# # image = np.asarray(result)
#
# def get_main_color(image):
#     image = np.array(image)
#     np.shape(image)[0]
#     image = image.reshape(-1, image.shape[-1])
#     m, _ = scipy.stats.mode(image, axis=0, keepdims=False)
#     return tuple(m)
#
#
#
# def get_augmented_views_gpu(images_tensor):
#     transform = transforms.RandomApply([
#                                         transforms.RandomHorizontalFlip(),
#                                         transforms.RandomAdjustSharpness(0.2),
#                                         transforms.RandomAdjustSharpness(2),
#                                         transforms.RandomPerspective(fill=(0, 0, 0)),
#                                         transforms.RandomRotation(degrees = 45, fill= (0, 0, 0)),
#                                        #  transforms.ColorJitter(brightness=0.1, contrast = 0.1, saturation=0.1, hue=0.0),
#                                        ],p=1)
#     res = transform(images_tensor.cuda()).cpu()
#     return res
#
#
