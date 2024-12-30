import torch 
from ultralytics import YOLO
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
import torchvision
import cv2
from torchvision.io.image import read_image
from utils.my_utils import OpenImageDataset, get_metrics
from PIL import Image
from torchvision.transforms import transforms, ToTensor
import numpy as np
from utils.my_utils import draw_boxes
import matplotlib.pyplot as plt
import copy 
import pytesseract
import os
import imutils
import json

class ALPR:

    def __init__(self):
        self.faster_weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.faster_rcnn = fasterrcnn_resnet50_fpn_v2(weights=self.faster_weights, box_score_thresh=0.55)
        self.faster_rcnn.eval()

        self.yolo8_car = YOLO(verbose=False)
        self.ssd_weights = torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1
        self.ssd = torchvision.models.detection.ssd300_vgg16(weights= self.ssd_weights)
        self.ssd.eval()

        self.yolo_plate = YOLO('models/license_plate_detector.pt')
        self.yolo_plate_seg = YOLO('models/best.pt',verbose=False)

        self.ocr_tesseract = pytesseract
        # self.ocr_tesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        config ='--oem 3 --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

       


        
        self.toPil = transforms.ToPILImage()
        self.toTensor = ToTensor()

    def process(self, images,vehicle_model="yolo",plate_task="bbox", output_dir="results"):
        out = []
        pil_images = []

        for i in images:
            image = Image.open(i)
            pil_images.append(image)
            


        if vehicle_model=="yolo":
            vehicles = self.use_yolo(pil_images)
        elif vehicle_model == "faster":
            vehicles = self.use_faster(pil_images)
        elif vehicle_model == "ssd":
            vehicles = self.use_ssd(pil_images )
        
        for i in range(len(vehicles)):
            image_detections = {'detections':[]}
            for y in range(len(vehicles[i]['boxes'])):
                detections = {'vehicle': vehicles[i]['boxes'][y].tolist(),
                              'class':vehicles[i]['labels'][y].tolist(),
                               'score': vehicles[i]['scores'][y].tolist()}
                image_detections['detections'].append(detections)
                # if plates[i]['labels'][y] == 0:
                #     detections['license_plate_bbox'] = None
                #     detections['text'] = None
                # else:
                #     detections['license_plate_bbox'] = plates[i]['boxes'][y].tolist()
                #     detections['text'] = plates[i]['labels'][y].tolist() 
            out.append(image_detections)



        if plate_task == "seg":
           self._detect_plates_seg(out,pil_images)
        else:
            self._detect_plates_bbox(out,pil_images)

        
        
        self._get_char(out,pil_images,plate_task)
        

        
        os.makedirs(output_dir,exist_ok=True)
        for i in range(len(images)):
            file_name = os.path.splitext(os.path.basename(images[i]))[0]
            with open(os.path.join(output_dir,file_name+".json"),'w') as f:
                
                json.dump(out[i],f,indent=4)


        # return vehicles, plates ,text, pre_pr, out
    
    def use_yolo(self,image):
        out = self.yolo8_car(image)
        out = self._get_yolo_predictions(out)
        out_filt = self._get_classes(out,[3,4,6,8])
        return out_filt
    
    def use_faster(self,images):
        with torch.no_grad():
            transform = self.faster_weights.transforms()
            images_transform = [transform(image) for image in images]
            out = self.faster_rcnn(images_transform)
            out_filt = self._get_classes(out,[3,4,6,8])
        
        return out_filt

    def use_ssd(self,images):
        with torch.no_grad():
            transform = self.ssd_weights.transforms()
            images_transform = [transform(image) for image in images]

            out = self.ssd(images_transform)
            out_filt = self._get_classes(out,[3,4,6,8])

            for i in out_filt:
                indices = np.where(i['scores']>=0.5)

                i['boxes'] = i['boxes'][indices]
                i['scores'] = i['scores'][indices]
                i['labels'] = i['labels'][indices]

        return out_filt
    
    def _detect_plates_bbox(self, predictions,images):
        images_list =[]
        
        for idx, pred in enumerate(predictions):
            image = images[idx]
            
            detections = pred["detections"]
            result_dict = {'boxes':[],
                'labels':[],
                'scores':[]}

            for detection in detections:
                box = detection['vehicle']
                image_cropped = image.crop((box[0],box[1],box[2],box[3]))
                pred =self.yolo_plate(image_cropped)
                for i in pred:
                    if len(i.boxes) > 0:
                        w = box[2] - box[0]
                        h = box[3] - box[1]
                        
                        license_plate=i.boxes.xyxy[0].numpy()
                        
                        x1box = license_plate[0] + box[0]
                        y1box = license_plate[1] + box[1] 
                        x2box = license_plate[2] + box[0]
                        y2box = license_plate[3] + box[1]
                        detection['license_plate_bbox'] = [float(x1box),float(y1box),float(x2box),float(y2box)]
                        # result_dict['labels'].append(i.boxes.cls.int().numpy()[0]+1)
                        # result_dict['scores'].append(i.boxes.conf.numpy()[0])
                        # result_dict['boxes'].append()
                    else:
                        detection['license_plate_bbox'] = None

                        # result_dict['boxes'].append(0)
                        # result_dict['scores'].append(0)
                        # result_dict['labels'].append(0)
            images_list.append(result_dict)
        return images_list
    
    def _detect_plates_seg(self, predictions,images):
        images_list =[]
        for idx, pred in enumerate(predictions):
            image = images[idx]
            detections = pred["detections"]
            result_dict = {'masks':[],
                'labels':[],
                'scores':[]}

            for detection in detections:
                box = detection['vehicle']

                image_cropped = image.crop((box[0],box[1],box[2],box[3]))
                x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
                predictions =self.yolo_plate_seg(image_cropped,task="segment")
                
                for pred_plate in predictions:
                    if pred_plate.masks is not None:
                        for i_mask in range(len(pred_plate.masks)):
                            
                            mask_xy = pred_plate.masks[i_mask].xy.pop() # The segmentation masks
                            # boxe = i.boxes[0]  
                            
                            mask_xy = mask_xy.astype(np.int32)
                            mask_xy = mask_xy.reshape(-1,1,2)
                            mask_xy = np.squeeze(mask_xy, axis=1)
                            mask_xy = mask_xy + [x1,y1]
                            detection['license_plate_mask'] = mask_xy.tolist()
                            break
                            # mask_image = cv2.drawContours(image, [mask], -1, (255, 255, 255), cv2.FILLED)
                            # result_dict['labels'].append(1)
                            # result_dict['scores'].append(pred_plate.boxes[i_mask].conf.numpy()[0])
                            # result_dict['masks'].append(mask_xy)
                    else:
                            detection['license_plate_mask'] = None



            images_list.append(result_dict)
        return images_list

    def _get_classes(self,outputs, classes, final_class=None):
        
        results = []
        for output in outputs:
            result ={}
            labels = output["labels"]
            indices = np.isin(labels, classes)
            
            for key in output.keys():
                if torch.is_tensor(output[key]):
                    result[key] = output[key].numpy()[indices]
                else:
                    result[key] = np.array(output[key])[indices]
                

            if final_class is not None:
                result["labels"][True] = final_class
            results.append(result)
        return results
    
    def _get_yolo_predictions(self, pred):
        out = []
        for i in pred:
            classes = i.boxes.cls.int() + 1
            out_dict = {
                "boxes": i.boxes.xyxy.tolist(),
                "labels": classes.tolist(),
                "scores": i.boxes.conf.tolist(),
            }
            out.append(out_dict)
        return out
    
    def _get_char(self,predictions,images,plate_task):
        images_list = []
        pre_pr = []

        if plate_task == "seg":
            type = "license_plate_mask"
        elif plate_task == "bbox":
            type = 'license_plate_bbox'

        for idx, pred in enumerate(predictions):
            image = images[idx]
            detections = pred['detections']
            plates_char = []
           
            for detection in detections:
                plate = detection[type]
                
                if plate is not None:
                    if plate_task=="seg":
                        
                        image_cropped = self._correct_orientation(np.array(plate),image)
                        
                        if len(image_cropped)>0:
                            pre_ = self._preprocess_ocr(np.array(image_cropped))
                            pre_pr.append(pre_)
                            pred =self.ocr_tesseract.image_to_string(pre_, config='--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                            # pred =self.ocr_tesseract.image_to_string(pre_, config='--oem 3 --psm 8')
                            detection['text']= pred
                            plates_char.append(pred)
                    else:
                        image_cropped = image.crop((plate[0],plate[1],plate[2],plate[3]))
                        
                        pre_ = self._preprocess_ocr(np.array(image_cropped))
                        pre_pr.append(pre_)
                        pred =self.ocr_tesseract.image_to_string(pre_, config='--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                        # pred =self.ocr_tesseract.image_to_string(pre_, config='--oem 3 --psm 8')
                        detection['text']= pred
                        print(pred)

                        plates_char.append(pred)
                        # images_list.append(new_image)
                else:
                    detection['text'] = None
        return plates_char,pre_pr
    
    def _correct_orientation(self,contour,image):
       
        x1, y2, w2, h2 = cv2.boundingRect(contour)
        image1 = np.array(image)
        cropped_image1 = image1[y2:y2+h2, x1:x1+w2][:, :, ::-1]
       
        
        # image = np.array(image)
        # rect = cv2.minAreaRect(contour)
        
        # box = cv2.boxPoints(rect)  # Get the four corners of the bounding box
        # box = np.int8(box) 
        # print(box)
        # angle = rect[2]

        # # Adjust angle based on width and height
        # if rect[1][0] < rect[1][1]:
        #     angle = angle - 90  # Correct if height is greater than width
        

        # (h, w) = image.shape[:2]
        # center = (w // 2, h // 2)

        # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        # rect_points_rotated = cv2.transform(box[None, :, :], rotation_matrix)
        
        
        # # cv2.namedWindow('Rotated image', cv2.WINDOW_NORMAL)
        # # cv2.resizeWindow('Rotated image', 800, 600)
        # # cv2.imshow('Rotated image',rotated_image)

        # # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # # cv2.resizeWindow('image', 800, 600)
        # # cv2.imshow('image',image)
        

        
        # # crop image
        # mask = np.zeros(rotated_image.shape[:2], dtype=np.uint8)
        # cv2.fillPoly(mask, [rect_points_rotated.astype(int)], 255)
        # cropped_image = cv2.bitwise_and(rotated_image, rotated_image, mask=mask)

        # cv2.namedWindow('cropped image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('cropped image', 800, 600)
        # cv2.imshow('cropped image',cropped_image)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # # x, y, w, h = cv2.boundingRect(rect_points_rotated)
        # # cropped_image = cropped_image[y:y+h, x:x+w]
        image = np.array(image)
        rect = cv2.minAreaRect(contour)
        
        box = cv2.boxPoints(rect)  # Get the four corners of the bounding box
        box = np.intp(box) 

        angle = rect[2]

        # Adjust angle based on width and height
        if rect[1][0] < rect[1][1]:
            angle = angle - 90  # Correct if height is greater than width
        
        print(angle)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        rect_points_rotated = cv2.transform(box[None, :, :], rotation_matrix)

        # crop image
        mask = np.zeros(rotated_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [rect_points_rotated.astype(int)], 255)
        cropped_image = cv2.bitwise_and(rotated_image, rotated_image, mask=mask)
        x, y, w, h = cv2.boundingRect(rect_points_rotated)
        cropped_image = cropped_image[y:y+h, x:x+w]


        # cv2.namedWindow('Rotated image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Rotated image', 800, 600)
        # cv2.imshow('Rotated image',rotated_image)

        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 800, 600)
        # cv2.imshow('image',cropped_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return cropped_image
        
    def _preprocess_ocr(self,img):
        # cv2.namedWindow('Rotated image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Rotated image', 800, 600)
        
        # cv2.imwrite('rotated.jpg',img[:, :, ::-1])

       
    
        # print(img.shape)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray,(5,5),0)
        # binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # # binary_image = cv2.adaptiveThreshold(
        # #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # processed_image = cv2.medianBlur(binary_image, 3)


        # scale_percent = 300  # percent of original size
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        # dim = (width, height)

        # resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
        # gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        # # Increase contrast by applying adaptive histogram equalization
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # gray = clahe.apply(gray)

        # # Blur the image slightly to remove noise
        # blur = cv2.GaussianBlur(gray, (5,5), 0)

        # # Use thresholding to make the image binary
        # _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # kernel = np.ones((3, 3), np.uint8)
        # close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        #------------------------------------------------------------------------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.namedWindow('Gray', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Gray', 800, 600)
        
        # cv2.imwrite('gray.jpg',gray)
        
       
       

        # Resize the image to improve OCR performance
        scale_percent = 600  # increase size by 2x
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_img = cv2.resize(gray, dim, interpolation=cv2.INTER_LINEAR)

        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_img = clahe.apply(resized_img)

        # Apply Gaussian Blur to remove noise
        blurred_img = cv2.GaussianBlur(contrast_img, (5, 5), 0)
        # print(blurred_img.mean())
        # cv2.namedWindow('Noise Removel', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Noise Removel', 800, 600)
        
        # cv2.imwrite('removal.jpg',blurred_img)

       
       
        # Use Otsu's Thresholding to binarize the image
        _, thresh_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # thresh_img = cv2.adaptiveThreshold(blurred_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #     cv2.THRESH_BINARY,5,1)

        # # plt.imshow(blurred_img,cmap='gray')
        # # plt.show()

        inverted_image = cv2.bitwise_not(thresh_img)
        # cv2.namedWindow('Noise Removel', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Noise Removel', 800, 600)
        
        # cv2.imwrite('bin.jpg',thresh_img)

       
        
        # # plt.imshow(inverted_image,cmap='gray')
        # # plt.show()
        # ------------------------------------------------------------------------------------------------

        # img = cv2.resize(img, (620,480) )
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.bilateralFilter(gray, 13, 15, 15)
        # edged = cv2.Canny(gray, 30, 200)
        # contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours = imutils.grab_contours(contours)
        # contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        # for c in contours:
        #     peri = cv2.arcLength(c, True)
        #     approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        #     if len(approx) == 4:
        #         screenCnt = approx
        #         break
        # if screenCnt is None:
        #     detected = 0
        #     print ("No contour detected")
        # else:
        #     detected = 1
        # if detected == 1:
        #     cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

        # mask = np.zeros(gray.shape,np.uint8)
        # new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
        # new_image = cv2.bitwise_and(img,img,mask=mask)

        # (x, y) = np.where(mask == 255)
        # (topx, topy) = (np.min(x), np.min(y))
        # (bottomx, bottomy) = (np.max(x), np.max(y))
        # Cropped = gray[topx:bottomx+1, topy:bottomy+1]

        # text = pytesseract.image_to_string(Cropped, config='--psm 11')
        return thresh_img


if __name__ == "__main__":
    # alpr = ALPR()

  
    # images = os.listdir("..\\downloaded_images")
    # images_full = [ os.path.join("../downloaded_images",i) for i in images]
    
    # alpr.process(images_full,vehicle_model="yolo",plate_task='seg',output_dir="new_results")

    folder_path = "results/new"

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if "new_results" in filename:  # Check if 'new_results' is in the file name
            new_filename = filename.replace("new_results", "")  # Remove 'new_results'
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

    print("All files with 'new_results' in their names have been renamed.")
