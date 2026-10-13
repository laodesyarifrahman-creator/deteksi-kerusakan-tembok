    from ultralytics import YOLO

    path = Path(input('Path model best.pt: ').strip().strip('"').strip("'"))
    if not path.is_file():
        raise FileNotFoundError(f'Model tidak ditemukan: {path}')
    source_text = input('Path video, atau nomor kamera (0/1): ').strip().strip('"').strip("'")
    camera = source_text.isdigit()
    source = int(source_text) if camera else str(Path(source_text))
    if not camera and not Path(source).is_file():
        raise FileNotFoundError(f'Video tidak ditemukan: {source}')
    model = YOLO(str(path))
    if model.task != 'detect':
        raise ValueError('Gunakan model YOLO detection / HBB.')
    print('Kelas model:', model.names)
    print('Q / Esc: keluar.')
    print(f'Membuka sumber: {source}', flush=True)
    if camera and os.name == 'nt':
        print('Backend webcam: DirectShow', flush=True)
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            print('DirectShow gagal; mencoba backend default...', flush=True)
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError('Video/kamera gagal dibuka. Untuk webcam coba indeks 0, 1, atau 2; tutup aplikasi kamera lain.')
        if camera:
            # Permintaan resolusi; kamera dapat memilih resolusi yang berbeda.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print('Sumber terbuka. Membaca frame pertama...', flush=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
            print('FPS sumber tidak valid; menggunakan 25 FPS sebagai perkiraan.')
        tracker = Stabilizer()
        index, prev_ts = 0, -1.0
        window = 'Deteksi Cacat Tembok - Kotak Stabil'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1000, 700)
        while True:
            start = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                if camera or index == 0:
                    raise RuntimeError('Sumber terbuka tetapi frame gagal dibaca. Coba indeks kamera lain dan periksa izin kamera Windows.')
                break
            if index == 0:
                print(f'Frame masuk: {frame.shape[1]}x{frame.shape[0]}. Memulai YOLO; inferensi pertama bisa lebih lama.', flush=True)
                cv2.imshow(window, frame)
                cv2.waitKey(1)
            if camera:
                timestamp = time.monotonic()
            else:
                pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                timestamp = pos if math.isfinite(pos) and pos > prev_ts else max(index/fps, prev_ts+1/fps)
            prev_ts = timestamp
            index += 1
            t0 = time.perf_counter()
            result = model.predict(frame, device='cpu', imgsz=IMGSZ,
                conf=CONF_PERTAHANKAN, iou=0.5, agnostic_nms=False, verbose=False)[0]
            infer_ms = (time.perf_counter()-t0)*1000
            rows = result.boxes.data.cpu().tolist() if result.boxes is not None else []
            tracks = tracker.update(rows, timestamp)
            for ident, t in tracks.items():
                x1, y1, x2, y2 = map(int, t['box'])
                color = ((0, 200, 0) if t['cls'] == 0 else (0, 70, 255)) if t['fresh'] else (0, 200, 255)
                label = f"{model.names[t['cls']]} #{ident} {t['conf']:.0%}"
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (max(0,x1), max(20,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            cv2.putText(frame, f'CPU | Inferensi {infer_ms:.0f} ms | Q: keluar',
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
            # Keterangan visual berdasarkan deteksi, bukan penilaian seluruh permukaan.
            fresh_defects = [t for t in tracks.values() if t['fresh']
                             and str(model.names[t['cls']]).lower() in ('retak', 'mengelupas')]
            held_defects = [t for t in tracks.values() if not t['fresh']
                            and str(model.names[t['cls']]).lower() in ('retak', 'mengelupas')]
            if fresh_defects:
                messages = ['PERLU PERBAIKAN', 'BELUM SIAP DICAT']
                banner_color = (0, 0, 220)
            elif held_defects:
                messages = ['PERLU PERBAIKAN', 'BELUM SIAP DICAT']
                banner_color = (0, 150, 200)
            else:
                messages = ['CACAT TIDAK TERDETEKSI', 'Kesiapan cat belum dinilai']
                banner_color = (80, 80, 80)
            height, width = frame.shape[:2]
            panel_height = min(70, height)
            cv2.rectangle(frame, (0, height-panel_height), (width, height), banner_color, -1)
            for line_no, message in enumerate(messages):
                base_width = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
                text_scale = min(0.6, 0.6 * max(1, width-20) / max(1, base_width))
                cv2.putText(frame, message, (10, height-panel_height+25+line_no*27),
                            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255,255,255), 2)
            cv2.imshow(window, frame)
            delay = 1 if camera else max(1, round(1000/fps-(time.perf_counter()-start)*1000))
            if cv2.waitKey(delay) & 0xFF in (ord('q'), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        if index == 0:
            raise RuntimeError('Tidak ada frame terbaca.')
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nDihentikan.')
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'\nERROR: {exc}', flush=True)
        raise SystemExit(1)
