# avoid_utils.py
import math
import rospy
import numpy as np
import pyproj

class UTILS_MISI_AVOID():
    def __init__(self):
        pass

    def kecepatan_belok(self, sudut, vel_y, vel_koef):
        if sudut > 60:
            sudut = 60
        elif sudut < -60:
            sudut = -60
        try:
            vel_z = math.sin(sudut) * vel_y * vel_koef
            return vel_z
        except Exception:
            return 0

    def cari_sudut(self, TITIK_TENGAH, TITIK_TENGAH_FRAME):
        if TITIK_TENGAH is None or TITIK_TENGAH_FRAME is None:
            return 0
        try:
            SUDUT = math.atan2(TITIK_TENGAH[0] - TITIK_TENGAH_FRAME[0], TITIK_TENGAH[1] - TITIK_TENGAH_FRAME[1])
            return SUDUT
        except Exception:
            return 0

    def pilih_bola_terdekat(self, kumpulan_bola):
        # safe: return None for empty/invalid input
        if not kumpulan_bola:
            return None
        try:
            # Expect each item to be a box or (score,..) depending on caller
            bola_tersortir = sorted(kumpulan_bola, key=lambda y: y[1] if (isinstance(y, (list, tuple)) and len(y) > 1) else 0, reverse=True)
            return bola_tersortir[0]
        except Exception:
            return None

    def cari_titik_tengah_bbox(self, box):
        if box is None:
            return None
        try:
            x1, y1, x2, y2 = box
            titik_tengah_bbox = (int((x1 + x2) / 2), int(y2))
            return titik_tengah_bbox
        except Exception:
            return None

    def titik_tengah_1_bola(self, BOLA, W, H, Y_ROI, reverse=False):
        if BOLA is None:
            return None
        try:
            p = BOLA
            C = np.interp(p[1],(Y_ROI, H),(W//4, W//2))
            if reverse:
                C *=-1
            x = p[0] + C
            x = max(0,min(x,W))
            titik_tengah = (x, p[1])
            return titik_tengah
        except Exception:
            return None

    def titik_tengah_2_bola(self, BOLA_1, BOLA_2):
        if BOLA_1 is None or BOLA_2 is None:
            return None
        try:
            p1 = BOLA_1
            p2 = BOLA_2
            titik_tengah = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            return titik_tengah
        except Exception:
            return None

    def titik_tengah_3_bola(self, BOLA_MERAH, BOLA_HITAM, BOLA_HIJAU):
        if BOLA_MERAH is None or BOLA_HITAM is None or BOLA_HIJAU is None:
            return None
        try:
            p1 = BOLA_MERAH
            p2 = BOLA_HITAM
            p3 = BOLA_HIJAU
            d21 = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
            d23 = (p2[0] - p3[0])**2 + (p2[1] - p3[1])**2

            if d21 > d23:
                titik_terjauh = p1
            elif d23 > d21:
                titik_terjauh = p3
            else:
                titik_terjauh = p1

            titik_tengah = ((p2[0] + titik_terjauh[0]) / 2, (p2[1] + titik_terjauh[1]) / 2)
            return titik_tengah
        except Exception:
            return None

    def cari_kecepatan_sudut(self, GAIN_YAW, TITIK_TENGAH, WIDTH):
        if TITIK_TENGAH is None or WIDTH is None or WIDTH == 0:
            return 0
        
        GAIN_LEFT = GAIN_YAW["LEFT"]
        GAIN_RIGHT = GAIN_YAW["RIGHT"]
        
        try:
            e = (2 * TITIK_TENGAH[0] - WIDTH) / WIDTH
            if TITIK_TENGAH[0] < WIDTH:
                kecepatan_sudut = GAIN_LEFT * e
                return (-1*kecepatan_sudut)
            else:
                kecepatan_sudut = GAIN_RIGHT * e
                return (-1*kecepatan_sudut)
        except Exception:
            return 0

    def kalkulasi_kecepatan_sudut(self, BOLA_MERAH, BOLA_HITAM, BOLA_HIJAU, GAIN_YAW, WIDTH, KONSTANTA):
        bola_terdeteksi = {
            "BOLA_MERAH": BOLA_MERAH,
            "BOLA_HITAM": BOLA_HITAM,
            "BOLA_HIJAU": BOLA_HIJAU
        }
        valid = {k: v for k, v in bola_terdeteksi.items() if v is not None}

        titik_tengah = None
        try:
            if len(valid) == 3:
                titik_tengah = self.titik_tengah_3_bola(valid.get("BOLA_MERAH"), valid.get("BOLA_HITAM"), valid.get("BOLA_HIJAU"))
            elif len(valid) == 2:
                bola = list(valid.values())
                titik_tengah = self.titik_tengah_2_bola(bola[0], bola[1])
            elif len(valid) == 1:
                bola = list(valid.values())
                titik_tengah = self.titik_tengah_1_bola(bola[0], KONSTANTA)
            else:
                rospy.loginfo("GERAK MENUJU WP--TIDAK MENDETEKSI BOLA")
                return None, titik_tengah
        except Exception:
            return None, titik_tengah

        kecepatan_sudut = self.cari_kecepatan_sudut(GAIN_YAW, titik_tengah, WIDTH)
        return kecepatan_sudut, titik_tengah

    # Navigasi LAMPU
    def titik_tuju(self, BOLA_HITAM, WIDTH, C):
        if BOLA_HITAM is None or WIDTH is None:
            return 0
        try:
            Bx, By = BOLA_HITAM
            arah = 1 if Bx <= WIDTH // 2 else -1

            y_min, y_max = 200, 360
            x_min, x_max = 0, WIDTH // 2
            x_konstan = 200  # fallback

            denom = (x_max - x_min) if (x_max - x_min) != 0 else 1
            m = (y_max - y_min) / denom
            y = min(max(By, y_min), y_max)
            x = (y - y_min + m * x_min) / (m if m != 0 else 1)
            x += arah * (WIDTH // 2)
            x = min(max(0, x), WIDTH)

            return (x+C) if y <= y_max else x_konstan
        except Exception:
            return 0

    def hitung_posisi_lampu(self, STATE_SEKARANG, JARAK_LAMPU):
        if STATE_SEKARANG is None:
            return None
        try:
            x_kapal, y_kapal, heading = STATE_SEKARANG
            arah_global_objek = (math.pi / 2) - heading
            x_lampu = x_kapal + JARAK_LAMPU * np.sin(arah_global_objek)
            y_lampu = y_kapal + JARAK_LAMPU * np.cos(arah_global_objek)
            return (x_lampu, y_lampu)
        except Exception:
            return None

    def buat_wp_mengitari_lampu(self, POSISI_LAMPU, JUMLAH_TITIK, origin, RADIUS):
        if POSISI_LAMPU is None:
            return []
        try:
            x_lampu, y_lampu = POSISI_LAMPU
            origin_lat, origin_lon = origin
            global_crs = pyproj.CRS("EPSG:4326")
            local_crs = pyproj.CRS(f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} +datum=WGS84")
            transformer = pyproj.Transformer.from_crs(local_crs, global_crs, always_xy=True)

            daftar_waypoint = []
            for i in range(JUMLAH_TITIK):
                sudut_titik = (2 * np.pi * i) / JUMLAH_TITIK
                wp_x = x_lampu + RADIUS * np.cos(sudut_titik)
                wp_y = y_lampu + RADIUS * np.sin(sudut_titik)
                wp_lon, wp_lat = transformer.transform(wp_x, wp_y)
                daftar_waypoint.append((wp_lat, wp_lon))

            return daftar_waypoint
        except Exception:
            return []

    def posisi_lampu_merah(self, STATE_SEKARANG, JARAK_LAMPU, origin):
        pos = self.hitung_posisi_lampu(STATE_SEKARANG, JARAK_LAMPU)
        if pos is None:
            return None
        try:
            x_lampu, y_lampu = pos
            origin_lat, origin_lon = origin
            global_crs = pyproj.CRS("EPSG:4326")
            local_crs = pyproj.CRS(f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} +datum=WGS84")
            transformer = pyproj.Transformer.from_crs(local_crs, global_crs, always_xy=True)
            x_g, y_g = transformer.transform(x_lampu, y_lampu)
            return (x_g, y_g)
        except Exception:
            return None
        
        
import math
import numpy as np

class OrbitController:
    """
    Vector Field Orbit Controller dengan tracking sudut 360° (CCW)
    """
    def __init__(self, center, radius, direction='ccw', k_orbit=1.0, k_rad=0.5):
        """
        Args:
            center: (x, y) pusat orbit dalam koordinat lokal (meter)
            radius: jarak orbit dari pusat (meter)
            direction: 'ccw' (counter-clockwise) atau 'cw' (clockwise)
            k_orbit: gain kecepatan tangensial
            k_rad: gain koreksi radial
        """
        self.center = center
        self.radius = radius
        self.direction = direction
        self.k_orbit = k_orbit
        self.k_rad = k_rad
        
        # Tracking sudut
        self.last_angle = None
        self.cumulative_angle = 0.0
        self.orbit_complete = False
        self.start_angle = None
        
    def reset(self):
        """Reset state orbit"""
        self.last_angle = None
        self.cumulative_angle = 0.0
        self.orbit_complete = False
        self.start_angle = None
        
    def _normalize_angle(self, angle):
        """Normalize angle ke range [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _angle_diff(self, new_angle, old_angle):
        """
        Hitung perbedaan sudut (new - old), normalized.
        CCW = negatif, CW = positif
        """
        diff = new_angle - old_angle
        return self._normalize_angle(diff)
    
    def update(self, pos, current_heading, forward_speed=1.0):
        """
        Update orbit controller dan return steering command.
        
        Args:
            pos: (x, y) posisi kapal dalam koordinat lokal (meter)
            current_heading: heading kapal saat ini (radian)
            forward_speed: kecepatan maju (m/s)
            
        Returns:
            (yaw_rate, is_complete, info_dict)
            - yaw_rate: kecepatan angular yang harus diberikan
            - is_complete: True jika sudah 360°
            - info_dict: informasi debug
        """
        if self.orbit_complete:
            return 0.0, True, {"cumulative_deg": 360.0, "status": "complete"}
        
        cx, cy = self.center
        px, py = pos
        
        # Vektor dari pusat ke kapal
        dx = px - cx
        dy = py - cy
        dist = math.sqrt(dx*dx + dy*dy) + 1e-6
        
        # Sudut saat ini relatif terhadap pusat
        current_angle = math.atan2(dy, dx)
        
        # Inisialisasi sudut awal
        if self.start_angle is None:
            self.start_angle = current_angle
            self.last_angle = current_angle
        
        # Hitung delta sudut
        delta_angle = self._angle_diff(current_angle, self.last_angle)
        
        # Akumulasi sudut (CCW = negatif untuk atan2 standard)
        self.cumulative_angle += delta_angle
        self.last_angle = current_angle
        
        # Cek apakah sudah 360° (2π)
        # CCW: cumulative_angle akan menjadi negatif
        # CW: cumulative_angle akan menjadi positif
        target_angle = -2 * math.pi if self.direction == 'ccw' else 2 * math.pi
        
        if self.direction == 'ccw':
            if self.cumulative_angle <= target_angle:
                self.orbit_complete = True
        else:
            if self.cumulative_angle >= target_angle:
                self.orbit_complete = True
        
        # === Vector Field Calculation ===
        # Unit vector radial (dari pusat ke kapal)
        ux, uy = dx / dist, dy / dist
        
        # Unit vector tangent (CCW: rotate 90° counter-clockwise)
        if self.direction == 'ccw':
            tx, ty = uy, -ux   # rotate -90° (CCW direction)
        else:
            tx, ty = -uy, ux   # rotate +90° (CW direction)
        
        # Error radial (positif = terlalu jauh dari pusat)
        radial_error = dist - self.radius
        
        # Desired velocity vector
        vx = self.k_orbit * tx - self.k_rad * radial_error * ux
        vy = self.k_orbit * ty - self.k_rad * radial_error * uy
        
        # Desired heading
        desired_heading = math.atan2(vy, vx)
        
        # Heading error
        heading_error = self._normalize_angle(desired_heading - current_heading)
        
        # Proportional yaw rate (bisa di-tune gain-nya)
        yaw_rate = 1.5 * heading_error
        
        # Clamp yaw rate
        max_yaw = 1.0  # rad/s
        yaw_rate = max(-max_yaw, min(max_yaw, yaw_rate))
        
        info = {
            "cumulative_deg": math.degrees(abs(self.cumulative_angle)),
            "radial_error": radial_error,
            "distance": dist,
            "current_angle_deg": math.degrees(current_angle),
            "status": "orbiting"
        }
        
        return yaw_rate, self.orbit_complete, info
