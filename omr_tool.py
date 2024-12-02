import os
import json
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import logging
from pathlib import Path
import time
from datetime import datetime

class OptikFormOkuyucu:
    """
    Optik form okuma işlemlerini gerçekleştiren sınıf.
    Görüntü işleme, referans noktaları bulma, hizalama, 
    işaretleri tespit etme ve yorumlama gibi işlemleri içerir.
    """
    
    SUPPORTED_FORMATS = {'.tiff', '.tif', '.jpg', '.jpeg', '.pdf'}
    REFERENCE_WIDTH = 1654
    REFERENCE_HEIGHT = 2339

    def __init__(self, json_file, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.debug_dir = Path('debug')

        # Klasörleri oluştur
        for dir_path in [self.output_dir, self.debug_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Log dosyasını ayarla
        self.log_file = self.debug_dir / f"optik_form_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.write_log("Program başlatıldı")

        # JSON konfigürasyonunu yükle
        self.json_file = json_file
        self.load_config()

    def write_log(self, message, filename=None):
        """Log mesajını dosyaya ve konsola yaz"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_message = f"{timestamp} - {message}"

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

        if filename:
            process_log = self.debug_dir / f"process_log_{Path(filename).stem}.txt"
            with open(process_log, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')

        print(log_message)

    def load_config(self):
        """JSON konfigürasyon dosyasını yükle"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.write_log(f"Konfigürasyon dosyası yüklendi: {self.json_file}")
        except Exception as e:
            self.write_log(f"HATA: Konfigürasyon dosyası yüklenemedi: {str(e)}")
            raise

    def detect_reference_points(self, image):
        """Form üzerindeki referans noktalarını tespit et"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Debug için eşiklenmiş görüntüyü kaydet
            cv2.imwrite(str(self.debug_dir / "thresh_reference.jpg"), thresh)

            # Köşelerdeki siyah kareleri bul
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            squares = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 100 < area < 400:  # Kare boyut aralığı
                    x, y, w, h = cv2.boundingRect(cnt)
                    if 0.9 < w/h < 1.1:  # Kare şekil kontrolü
                        squares.append((x + w//2, y + h//2))

                        # Debug için bulunan kareleri görselleştir
                        debug_img = image.copy()
                        cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0,255,0), 2)
                        cv2.imwrite(str(self.debug_dir / f"reference_square_{len(squares)}.jpg"), debug_img)

            self.write_log(f"Bulunan referans nokta sayısı: {len(squares)}")
            return squares

        except Exception as e:
            self.write_log(f"HATA: Referans noktaları tespit edilemedi: {str(e)}")
            raise

    def sort_reference_points(self, points):
        """Referans noktalarını sırala (sol üst, sağ üst, sol alt, sağ alt)"""
        points = np.array(points)
        center = points.mean(axis=0)
        
        # Noktaları çeyrek bölgelere göre sırala
        sorted_points = []
        for quarter in [(False,False), (True,False), (False,True), (True,True)]:
            mask = (points[:,0] > center[0]) == quarter[0]
            mask &= (points[:,1] > center[1]) == quarter[1]
            quarter_points = points[mask]
            if len(quarter_points) > 0:
                sorted_points.append(quarter_points[0])
        
        return np.array(sorted_points)

def align_form(self, image, reference_points):
    """Formu referans noktalarına göre hizala"""
    try:
        if len(reference_points) < 4:
            self.write_log("UYARI: Yeterli referans noktası bulunamadı, tahmini noktalar kullanılıyor")
        
        # Referans noktalarını sırala
        sorted_points = self.sort_reference_points(reference_points)
        
        # Hedef koordinatlar
        dst_points = np.float32([
            [0, 0],
            [self.REFERENCE_WIDTH, 0],
            [0, self.REFERENCE_HEIGHT],
            [self.REFERENCE_WIDTH, self.REFERENCE_HEIGHT]
        ])
        
        # Perspektif dönüşüm matrisi
        matrix = cv2.getPerspectiveTransform(
            np.float32(sorted_points),
            dst_points
        )
        
        # Görüntüyü düzelt
        aligned = cv2.warpPerspective(image, matrix, 
                                    (self.REFERENCE_WIDTH, self.REFERENCE_HEIGHT))
        
        # Debug için düzeltilmiş görüntüyü kaydet
        debug_img = image.copy()
        for i, (x,y) in enumerate(sorted_points):
            cv2.circle(debug_img, (int(x),int(y)), 5, (0,255,0), -1)
            cv2.putText(debug_img, str(i+1), (int(x)+10,int(y)+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imwrite(str(self.debug_dir / "reference_points.jpg"), debug_img)
        cv2.imwrite(str(self.debug_dir / "aligned_form.jpg"), aligned)
        
        return aligned
        
    except Exception as e:
        self.write_log(f"HATA: Form hizalama başarısız: {str(e)}")
        raise
        """Formu referans noktalarına göre hizala"""
        try:
            if len(reference_points) < 4:
                raise Exception(f"Yeterli referans noktası bulunamadı: {len(reference_points)}")
            
            # Referans noktalarını sırala
            sorted_points = self.sort_reference_points(reference_points)
            
            # Hedef koordinatlar
            dst_points = np.float32([
                [0, 0],
                [self.REFERENCE_WIDTH, 0],
                [0, self.REFERENCE_HEIGHT],
                [self.REFERENCE_WIDTH, self.REFERENCE_HEIGHT]
            ])
            
            # Perspektif dönüşüm matrisi
            matrix = cv2.getPerspectiveTransform(
                np.float32(sorted_points),
                dst_points
            )
            
            # Görüntüyü düzelt
            aligned = cv2.warpPerspective(image, matrix, 
                                        (self.REFERENCE_WIDTH, self.REFERENCE_HEIGHT))
            
            # Debug için düzeltilmiş görüntüyü kaydet
            cv2.imwrite(str(self.debug_dir / "aligned_form.jpg"), aligned)
            
            return aligned
            
        except Exception as e:
            self.write_log(f"HATA: Form hizalama başarısız: {str(e)}")
            raise

    def preprocess_image(self, image):
        """Görüntü ön işleme"""
        try:
            # Gri tonlamaya çevir
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Gürültü azaltma
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Adaptif eşikleme
            thresh = cv2.adaptiveThreshold(
                blurred, 
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                21,
                5
            )
            
            # Morfolojik işlemler
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            return cleaned
            
        except Exception as e:
            self.write_log(f"HATA: Görüntü ön işleme başarısız: {str(e)}")
            raise

    def detect_marks(self, roi, area_config):
        """İşaretli daireleri tespit et"""
        try:
            # Görüntüyü hazırla
            processed = self.preprocess_image(roi)
            
            # Grid boyutlarını al
            rows, cols = map(int, area_config['daire_sayısı'].split('x'))
            
            # Her alan tipi için farklı eşik değerleri
            thresholds = {
                'OKUL KODU': 0.35,
                'TC KİMLİK/CEP TEL': 0.35,
                'TEST': 0.25,
                'DEFAULT': 0.3
            }
            
            # Hücre boyutlarını hesapla
            cell_height = roi.shape[0] // rows
            cell_width = roi.shape[1] // cols
            
            marks = []
            for row in range(rows):
                for col in range(cols):
                    # Hücre bölgesini al
                    top = row * cell_height + int(cell_height * 0.2)
                    bottom = (row + 1) * cell_height - int(cell_height * 0.2)
                    left = col * cell_width + int(cell_width * 0.2)
                    right = (col + 1) * cell_width - int(cell_width * 0.2)
                    
                    cell = processed[top:bottom, left:right]
                    
                    # İşaretleme oranını hesapla
                    dark_pixel_ratio = np.sum(cell > 0) / cell.size
                    
                    # Alan tipine göre eşik değeri seç
                    area_type = area_config.get('type', 'DEFAULT')
                    threshold = thresholds.get(area_type, thresholds['DEFAULT'])
                    
                    # Debug bilgisi
                    self.write_log(f"{area_config['alan_adi']} - Row:{row} Col:{col} Ratio:{dark_pixel_ratio:.3f}")
                    
                    if dark_pixel_ratio > threshold:
                        marks.append((row, col))
                        
                        # İşaretli hücre görüntüsü
                        debug_cell = roi[row*cell_height:(row+1)*cell_height, 
                                       col*cell_width:(col+1)*cell_width].copy()
                        cv2.rectangle(debug_cell, (0,0), (cell_width-1,cell_height-1), (0,255,0), 2)
                        debug_path = self.debug_dir / f"marked_{area_config['alan_adi']}_{row}_{col}.jpg"
                        cv2.imwrite(str(debug_path), debug_cell)
            
            return marks
            
        except Exception as e:
            self.write_log(f"HATA: İşaret tespiti başarısız: {str(e)}")
            raise

    def interpret_marks(self, marks, area_config):
        """İşaretleri yorumla ve değerlere dönüştür"""
        try:
            if not marks:
                return ""
                
            if 'değerler' in area_config:
                if area_config['alan_adi'].startswith('TEST_'):
                    # Test cevapları için özel format
                    question_count = int(area_config['daire_sayısı'].split('x')[1])
                    answers = ['_'] * question_count
                    for row, col in sorted(marks):
                        if row < question_count:
                            answers[row] = 'ABCDE'[col]
                    return answers
                    
                elif area_config['alan_adi'] in ['OKUL KODU', 'TC KİMLİK/CEP TEL']:
                    # Sayısal alanlar için sıralı işleme
                    values = area_config['değerler']['sütunlar'].split(',')
                    sorted_marks = sorted(marks, key=lambda x: (x[1], x[0]))
                    result = ''.join(values[m[0]] for m in sorted_marks)
                    return result
                    
                elif 'karakterler' in area_config['değerler']:
                    # Karakter alanları için işleme
                    values = area_config['değerler']['karakterler'].split(',')
                    result = ''.join(values[m[1]] if m[1] < len(values) else ' ' 
                                   for m in sorted(marks))
                    return result
                    
                elif 'seçenekler' in area_config['değerler']:
                    # Tekli seçim alanları
                    values = area_config['değerler']['seçenekler'].split(',')
                    return values[marks[0][1]] if marks else ''
            
            return marks
            
        except Exception as e:
            self.write_log(f"HATA: İşaret yorumlama başarısız: {str(e)}")
            return ""

    def process_form(self, image_path):
        """Form işleme ana fonksiyonu"""
        try:
            self.write_log(f"\nForm işleniyor: {image_path}")
            
            # Görüntüyü oku
            image = cv2.imread(str(image_path))
            if image is None:
                raise Exception("Görüntü okunamadı")
            
            # Referans noktalarını bul
            reference_points = self.detect_reference_points(image)
            
            # Formu düzelt
            aligned_image = self.align_form(image, reference_points)
            
            # Her alanı işle
            results = {}
            for area_name, area_config in self.config.items():
                try:
                    # Alan koordinatlarını al
                    x1, y1 = area_config['başlangıç']
                    x2, y2 = area_config['bitiş']
                    
                    # Alanı kes
                    roi = aligned_image[y1:y2, x1:x2]
                    
                    # Debug için alanı kaydet
                    cv2.imwrite(str(self.debug_dir / f"roi_{area_name}.jpg"), roi)
                    
                    # Alan bilgilerini ekle
                    area_config['alan_adi'] = area_name
                    
                    # İşaretleri tespit et ve yorumla
                    marks = self.detect_marks(roi, area_config)
                    result = self.interpret_marks(marks, area_config)
                    
                    results[area_name] = result
                    self.write_log(f"{area_name}: {result}")
                    
                except Exception as e:
                    self.write_log(f"HATA: {area_name} alanı işlenirken hata: {str(e)}")
                    results[area_name] = None
            
            # Sonuçları kaydet
            output_path = self.output_dir / f"{Path(image_path).stem}_sonuc.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                for area_name, result in results.items():
                    if isinstance(result, list):
                        f.write(f"{area_name}: [{','.join(result)}]\n")
                    else:
                        f.write(f"{area_name}: {result}\n")
            
            return results
            
        except Exception as e:
            self.write_log(f"HATA: Form işlenirken hata: {str(e)}")
            raise

    def process_directory(self):
        """Klasördeki tüm dosyaları işle"""
        try:
            start_time = time.time()
            self.write_log("\nKlasör işleme başladı")
            
            files = [f for f in self.input_dir.iterdir() 
                    if f.suffix.lower() in self.SUPPORTED_FORMATS]
            
            self.write_log(f"İşlenecek dosya sayısı: {len(files)}")
            
            for file_path in files:
                try:
                    self.process_form(file_path)
                except Exception as e:
                    self.write_log(f"HATA: {file_path} işlenirken hata: {str(e)}")
                    continue
            
            elapsed_time = time.time() - start_time
            self.write_log(f"\nKlasör işleme tamamlandı. Süre: {elapsed_time:.2f} saniye")
            
        except Exception as e:
            self.write_log(f"HATA: Klasör işlenirken hata: {str(e)}")
            raise

def main():
    try:
        # Konfigürasyon
        json_file = "isler.json"
        input_dir = "data/optik_formlar"
        output_dir = "data/sonuclar"
        
        # İşleyiciyi oluştur
        processor = OptikFormOkuyucu(json_file, input_dir, output_dir)
        
        # Klasörü işle
        processor.process_directory()
        
    except Exception as e:
        print(f"Program çalışırken hata oluştu: {str(e)}")
        raise

if __name__ == "__main__":
    main()