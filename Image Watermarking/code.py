import cv2
import numpy as np
import pywt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from skimage.metrics import structural_similarity as ssim


class ReverseWatermarkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reverse Watermarking System")
        self.root.geometry("600x550")

        tk.Label(root, text="Reverse Watermarking System", font=("Arial", 14, "bold")).pack(pady=10)

        tk.Button(root, text="Select Cover Image", command=self.load_cover_image).pack(pady=5)
        tk.Button(root, text="Select Watermark Image", command=self.load_watermark).pack(pady=5)
        tk.Button(root, text="Embed Watermark", command=self.embed_watermark).pack(pady=5)
        tk.Button(root, text="Extract Watermark", command=self.extract_watermark).pack(pady=5)
        tk.Button(root, text="Verify Integrity", command=self.verify_watermark).pack(pady=5)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

    def load_cover_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
        if file_path:
            self.cover_image_path = file_path
            self.display_image(file_path)

    def load_watermark(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
        if file_path:
            self.watermark_path = file_path
            self.display_image(file_path)

    def embed_watermark(self):
        if not hasattr(self, 'cover_image_path') or not hasattr(self, 'watermark_path'):
            messagebox.showerror("Error", "Select Cover and Watermark Images First")
            return

        cover_img = cv2.imread(self.cover_image_path, cv2.IMREAD_COLOR)
        watermark = cv2.imread(self.watermark_path, cv2.IMREAD_COLOR)

        # Resize watermark to match cover image
        watermark = cv2.resize(watermark, (cover_img.shape[1], cover_img.shape[0]))

        # Split into R, G, B channels
        cover_r, cover_g, cover_b = cv2.split(cover_img)
        wm_r, wm_g, wm_b = cv2.split(watermark)

        alpha = 0.1  # Embedding strength

        def embed_dwt(cover_channel, wm_channel):
            coeffs_cover = pywt.dwt2(cover_channel, 'haar')
            LL, (LH, HL, HH) = coeffs_cover
            wm_resized = cv2.resize(wm_channel, (LL.shape[1], LL.shape[0]))
            LL_watermarked = LL + alpha * wm_resized
            return pywt.idwt2((LL_watermarked, (LH, HL, HH)), 'haar')

        # Embed watermark in each channel
        watermarked_r = embed_dwt(cover_r, wm_r)
        watermarked_g = embed_dwt(cover_g, wm_g)
        watermarked_b = embed_dwt(cover_b, wm_b)

        # Merge channels back to get the final color image
        watermarked_img = cv2.merge([np.uint8(np.clip(watermarked_r, 0, 255)), 
                                     np.uint8(np.clip(watermarked_g, 0, 255)), 
                                     np.uint8(np.clip(watermarked_b, 0, 255))])

        cv2.imwrite("watermarked_image.jpg", watermarked_img)
        messagebox.showinfo("Success", "Watermark Embedded Successfully!")
        self.display_image("watermarked_image.jpg")

    def extract_watermark(self):
        if not hasattr(self, 'cover_image_path'):
            messagebox.showerror("Error", "Embed Watermark First!")
            return

        watermarked_img = cv2.imread("watermarked_image.jpg", cv2.IMREAD_COLOR)
        cover_img = cv2.imread(self.cover_image_path, cv2.IMREAD_COLOR)

        if watermarked_img is None or cover_img is None:
            messagebox.showerror("Error", "Watermarked image not found!")
            return

        # Split into R, G, B channels
        wm_r, wm_g, wm_b = cv2.split(watermarked_img)
        cover_r, cover_g, cover_b = cv2.split(cover_img)

        alpha = 0.1

        def extract_dwt(wm_channel, cover_channel):
            coeffs_wm = pywt.dwt2(wm_channel, 'haar')
            LL_wm, (LH_wm, HL_wm, HH_wm) = coeffs_wm
            coeffs_cover = pywt.dwt2(cover_channel, 'haar')
            LL_cover, (_, _, _) = coeffs_cover
            extracted_wm = (LL_wm - LL_cover) / alpha
            return cv2.resize(np.uint8(np.clip(extracted_wm, 0, 255)), (cover_channel.shape[1], cover_channel.shape[0]))

        # Extract watermark from each channel
        extracted_r = extract_dwt(wm_r, cover_r)
        extracted_g = extract_dwt(wm_g, cover_g)
        extracted_b = extract_dwt(wm_b, cover_b)

        # Merge extracted R, G, B channels to reconstruct color watermark
        extracted_watermark = cv2.merge([extracted_r, extracted_g, extracted_b])
        cv2.imwrite("extracted_watermark.jpg", extracted_watermark)

        messagebox.showinfo("Success", "Watermark Extracted!")
        self.display_image("extracted_watermark.jpg")

    def verify_watermark(self):
        if not hasattr(self, 'watermark_path'):
            messagebox.showerror("Error", "Select Original Watermark First!")
            return
        
        extracted_path = "extracted_watermark.jpg"
        if cv2.imread(extracted_path) is None:
            smessagebox.showerror("Error", "Extract Watermark First!")
            return
        
        # Load original and extracted watermarks in color
        original_wm = cv2.imread(self.watermark_path, cv2.IMREAD_COLOR)
        extracted_wm = cv2.imread(extracted_path, cv2.IMREAD_COLOR)

        # Resize extracted watermark to match original
        extracted_wm = cv2.resize(extracted_wm, (original_wm.shape[1], original_wm.shape[0]))

        # Compute SSIM for each color channel
        ssim_r = ssim(original_wm[:, :, 0], extracted_wm[:, :, 0], data_range=255)
        ssim_g = ssim(original_wm[:, :, 1], extracted_wm[:, :, 1], data_range=255)
        ssim_b = ssim(original_wm[:, :, 2], extracted_wm[:, :, 2], data_range=255)

        avg_ssim = (ssim_r + ssim_g + ssim_b) / 3  # Average across R, G, B

        # Compute PSNR
        mse = np.mean((original_wm - extracted_wm) ** 2)
        psnr_value = 100 if mse == 0 else 10 * np.log10(255**2 / mse)

        if avg_ssim > 0.75:  # High similarity threshold
            messagebox.showinfo("Verification", f"Watermark Verified ✅\nSSIM: {avg_ssim:.4f}\nPSNR: {psnr_value:.2f} dB")
        else:
            messagebox.showerror("Verification", f"Watermark Tampered ❌\nSSIM: {avg_ssim:.4f}\nPSNR: {psnr_value:.2f} dB")

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img


# Run GUI
root = tk.Tk()
app = ReverseWatermarkingApp(root)
root.mainloop()
