#!/usr/bin/env python3
"""
Sentinel-2 Image Finder — GUI
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path


class _LogRedirect:
    def __init__(self, widget):
        self.widget = widget
    def write(self, text):
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")
        self.widget.update_idletasks()
    def flush(self):
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentinel-2 Image Finder")
        self.resizable(True, True)
        self.minsize(600, 600)
        self._cancel_event = None
        self._build_ui()
        self._load_env()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # ---- Field boundary ----
        sf_frame = ttk.LabelFrame(self, text="Field boundary  (.zip / .shp / .gpkg)")
        sf_frame.pack(fill="x", **pad)
        self.shapefile_var = tk.StringVar()
        ttk.Entry(sf_frame, textvariable=self.shapefile_var).pack(
            side="left", padx=(8, 4), pady=6, fill="x", expand=True)
        ttk.Button(sf_frame, text="Browse…", command=self._browse_shapefile).pack(
            side="left", padx=(0, 8), pady=6)

        # ---- Output folder ----
        out_frame = ttk.LabelFrame(self, text="Output Folder")
        out_frame.pack(fill="x", **pad)
        self.output_var = tk.StringVar(value=str(Path.cwd() / "output"))
        ttk.Entry(out_frame, textvariable=self.output_var).pack(
            side="left", padx=(8, 4), pady=6, fill="x", expand=True)
        ttk.Button(out_frame, text="Browse…", command=self._browse_output).pack(
            side="left", padx=(0, 8), pady=6)

        # ---- Settings ----
        settings_frame = ttk.LabelFrame(self, text="Settings")
        settings_frame.pack(fill="x", **pad)

        ttk.Label(settings_frame, text="Max cloud cover (%):").grid(
            row=0, column=0, sticky="e", padx=(8, 4), pady=4)
        self.cloud_var = tk.DoubleVar(value=10.0)
        ttk.Spinbox(settings_frame, from_=0, to=100, increment=1,
                    textvariable=self.cloud_var, width=6).grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(settings_frame, text="Parallel workers:").grid(
            row=0, column=2, sticky="e", padx=(16, 4), pady=4)
        self.workers_var = tk.IntVar(value=4)
        ttk.Spinbox(settings_frame, from_=1, to=10, increment=1,
                    textvariable=self.workers_var, width=4).grid(
                    row=0, column=3, sticky="w", pady=4, padx=(0, 8))

        ttk.Label(settings_frame, text="Output type:").grid(
            row=1, column=0, sticky="e", padx=(8, 4), pady=4)
        self.do_rgb_var         = tk.BooleanVar(value=True)
        self.do_ndwi_var        = tk.BooleanVar(value=False)
        self.do_ndvi_var        = tk.BooleanVar(value=False)
        self.do_false_color_var = tk.BooleanVar(value=False)
        self.do_dem_var         = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="RGB",
                        variable=self.do_rgb_var).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Checkbutton(settings_frame, text="NDWI (water index)",
                        variable=self.do_ndwi_var).grid(row=1, column=2, sticky="w", pady=4)
        ttk.Checkbutton(settings_frame, text="False Color (NIR)",
                        variable=self.do_false_color_var).grid(
                        row=1, column=3, sticky="w", pady=4, padx=(0, 8))
        ttk.Checkbutton(settings_frame, text="NDVI (vegetation health)",
                        variable=self.do_ndvi_var).grid(row=2, column=1, sticky="w", pady=(0, 6))
        ttk.Checkbutton(settings_frame, text="Elevation (DEM)",
                        variable=self.do_dem_var).grid(row=2, column=2, sticky="w", pady=(0, 6))

        # ---- Credentials ----
        cred_frame = ttk.LabelFrame(self, text="Copernicus Credentials")
        cred_frame.pack(fill="x", **pad)

        ttk.Label(cred_frame, text="Email:").grid(row=0, column=0, sticky="e", padx=(8, 4), pady=4)
        self.user_var = tk.StringVar()
        ttk.Entry(cred_frame, textvariable=self.user_var, width=34).grid(
            row=0, column=1, sticky="w", pady=4)

        ttk.Label(cred_frame, text="Password:").grid(row=1, column=0, sticky="e", padx=(8, 4), pady=4)
        self.pass_var = tk.StringVar()
        ttk.Entry(cred_frame, textvariable=self.pass_var, show="*", width=34).grid(
            row=1, column=1, sticky="w", pady=4)

        # ---- S3 credentials (optional) ----
        s3_frame = ttk.LabelFrame(self, text="S3 Credentials  (optional — faster downloads)")
        s3_frame.pack(fill="x", **pad)

        ttk.Label(s3_frame, text="S3 Key:").grid(row=0, column=0, sticky="e", padx=(8, 4), pady=4)
        self.s3key_var = tk.StringVar()
        ttk.Entry(s3_frame, textvariable=self.s3key_var, width=34).grid(
            row=0, column=1, sticky="w", pady=4)

        ttk.Label(s3_frame, text="S3 Secret:").grid(row=1, column=0, sticky="e", padx=(8, 4), pady=4)
        self.s3secret_var = tk.StringVar()
        ttk.Entry(s3_frame, textvariable=self.s3secret_var, show="*", width=34).grid(
            row=1, column=1, sticky="w", pady=4)

        ttk.Label(s3_frame, text="Generate at dataspace.copernicus.eu → Account → S3 credentials",
                  foreground="gray").grid(row=2, column=0, columnspan=2, sticky="w",
                                          padx=(8, 4), pady=(0, 4))

        # ---- OpenTopography ----
        ot_frame = ttk.LabelFrame(self, text="OpenTopography API Key  (optional — higher resolution DEM)")
        ot_frame.pack(fill="x", **pad)
        ttk.Label(ot_frame, text="API Key:").grid(row=0, column=0, sticky="e", padx=(8, 4), pady=4)
        self.ot_key_var = tk.StringVar()
        ttk.Entry(ot_frame, textvariable=self.ot_key_var, width=34).grid(
            row=0, column=1, sticky="w", pady=4)
        ttk.Label(ot_frame, text="Free key at opentopography.org/myopentopo",
                  foreground="gray").grid(row=1, column=0, columnspan=2, sticky="w",
                                          padx=(8, 4), pady=(0, 4))

        # ---- Run / Cancel ----
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=(8, 2))
        self.run_btn    = ttk.Button(btn_frame, text="Download Images", command=self._run)
        self.run_btn.pack(side="left", padx=(0, 6))
        self.cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self._cancel,
                                     state="disabled")
        self.cancel_btn.pack(side="left")

        # ---- Historical NDVI Average ----
        hist_frame = ttk.LabelFrame(self, text="Historical NDVI Average  (peak vegetation composite)")
        hist_frame.pack(fill="x", **pad)

        ttk.Label(hist_frame, text="Window start (M/D):").grid(
            row=0, column=0, sticky="e", padx=(8, 4), pady=4)
        self.hist_smonth_var = tk.IntVar(value=6)
        self.hist_sday_var   = tk.IntVar(value=15)
        ttk.Spinbox(hist_frame, from_=1, to=12, textvariable=self.hist_smonth_var,
                    width=4).grid(row=0, column=1, pady=4)
        ttk.Spinbox(hist_frame, from_=1, to=31, textvariable=self.hist_sday_var,
                    width=4).grid(row=0, column=2, pady=4, padx=(2, 0))

        ttk.Label(hist_frame, text="Window end (M/D):").grid(
            row=0, column=3, sticky="e", padx=(16, 4), pady=4)
        self.hist_emonth_var = tk.IntVar(value=8)
        self.hist_eday_var   = tk.IntVar(value=15)
        ttk.Spinbox(hist_frame, from_=1, to=12, textvariable=self.hist_emonth_var,
                    width=4).grid(row=0, column=4, pady=4)
        ttk.Spinbox(hist_frame, from_=1, to=31, textvariable=self.hist_eday_var,
                    width=4).grid(row=0, column=5, pady=4, padx=(2, 0))

        self.hist_btn = ttk.Button(hist_frame, text="Compute Historical Average",
                                   command=self._run_historical)
        self.hist_btn.grid(row=0, column=6, padx=(16, 8), pady=4)

        # ---- Progress ----
        prog_frame = ttk.Frame(self)
        prog_frame.pack(fill="x", padx=10, pady=(2, 4))
        self.progress = ttk.Progressbar(prog_frame, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True)
        self.progress_lbl = ttk.Label(prog_frame, text="", width=12, anchor="e")
        self.progress_lbl.pack(side="left", padx=(6, 0))

        # ---- Log ----
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = scrolledtext.ScrolledText(log_frame, state="disabled", height=10, wrap="word")
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

        sys.stdout = _LogRedirect(self.log)
        sys.stderr = _LogRedirect(self.log)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_env(self):
        env_path = Path(__file__).parent / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("CDSE_USERNAME="):
                self.user_var.set(line.split("=", 1)[1])
            elif line.startswith("CDSE_PASSWORD="):
                self.pass_var.set(line.split("=", 1)[1])
            elif line.startswith("CDSE_S3_KEY="):
                self.s3key_var.set(line.split("=", 1)[1])
            elif line.startswith("CDSE_S3_SECRET="):
                self.s3secret_var.set(line.split("=", 1)[1])
            elif line.startswith("OPENTOPO_KEY="):
                self.ot_key_var.set(line.split("=", 1)[1])

    def _browse_shapefile(self):
        path = filedialog.askopenfilename(
            title="Select field boundary",
            filetypes=[
                ("All supported", "*.zip *.gpkg *.shp"),
                ("GeoPackage",    "*.gpkg"),
                ("Zipped shapefile", "*.zip"),
                ("Shapefile",     "*.shp"),
                ("All files",     "*.*"),
            ],
        )
        if path:
            self.shapefile_var.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    # ------------------------------------------------------------------
    # Run / Cancel / Progress
    # ------------------------------------------------------------------

    def _run(self):
        shapefile = self.shapefile_var.get().strip()
        output    = self.output_var.get().strip()
        cloud     = self.cloud_var.get()
        workers   = self.workers_var.get()
        do_rgb    = self.do_rgb_var.get()
        do_ndwi   = self.do_ndwi_var.get()
        do_ndvi   = self.do_ndvi_var.get()
        do_fc     = self.do_false_color_var.get()
        do_dem    = self.do_dem_var.get()
        username  = self.user_var.get().strip()
        password  = self.pass_var.get()
        s3key     = self.s3key_var.get().strip()
        s3secret  = self.s3secret_var.get().strip()

        if not shapefile:
            print("ERROR: Please select a field boundary file (.zip, .shp, or .gpkg).")
            return
        if not do_rgb and not do_ndwi and not do_ndvi and not do_fc and not do_dem:
            print("ERROR: Select at least one output type.")
            return
        if not username or not password:
            print("ERROR: Please enter your Copernicus credentials.")
            return

        os.environ["CDSE_USERNAME"] = username
        os.environ["CDSE_PASSWORD"] = password
        if s3key and s3secret:
            os.environ["CDSE_S3_KEY"]    = s3key
            os.environ["CDSE_S3_SECRET"] = s3secret
        else:
            os.environ.pop("CDSE_S3_KEY",    None)
            os.environ.pop("CDSE_S3_SECRET", None)

        ot_key = self.ot_key_var.get().strip()
        if ot_key:
            os.environ["OPENTOPO_KEY"] = ot_key
        else:
            os.environ.pop("OPENTOPO_KEY", None)

        self._cancel_event = threading.Event()
        self.run_btn.configure(state="disabled")
        self.hist_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)
        self.progress_lbl.configure(text="Searching…")

        threading.Thread(
            target=self._worker,
            args=(shapefile, output, cloud, workers, do_rgb, do_ndwi, do_ndvi, do_fc, do_dem),
            daemon=True,
        ).start()

    def _run_historical(self):
        output = self.output_var.get().strip()
        if not output:
            print("ERROR: Output folder not set.")
            return
        ndvi_dir = Path(output) / "NDVI"
        if not ndvi_dir.exists():
            print("ERROR: NDVI subfolder not found. Run an NDVI download first.")
            return
        raw_count = len(list(ndvi_dir.glob("*_NDVI_raw.tif")))
        if raw_count == 0:
            old = len(list(ndvi_dir.glob("*_NDVI.tif")))
            if old > 0:
                print(f"ERROR: Found {old} colorized NDVI files but no raw rasters.")
                print("       Re-run NDVI download to populate *_NDVI_raw.tif files.")
            else:
                print("ERROR: No NDVI files at all. Run NDVI download first.")
            return

        smonth = self.hist_smonth_var.get()
        sday   = self.hist_sday_var.get()
        emonth = self.hist_emonth_var.get()
        eday   = self.hist_eday_var.get()

        self._cancel_event = threading.Event()
        self.run_btn.configure(state="disabled")
        self.hist_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)
        self.progress_lbl.configure(text="Aggregating…")

        threading.Thread(
            target=self._worker_historical,
            args=(ndvi_dir, smonth, sday, emonth, eday),
            daemon=True,
        ).start()

    def _worker_historical(self, ndvi_dir, smonth, sday, emonth, eday):
        try:
            import importlib
            import finder
            importlib.reload(finder)
            result = finder.compute_ndvi_historical(
                ndvi_dir, smonth, sday, emonth, eday,
                cancel_event=self._cancel_event,
                progress_callback=self._on_progress,
            )
            if result.get("needs_rerun"):
                print(f"\nNo raw NDVI files found "
                      f"({result.get('old_colorized', 0)} old colorized files present).")
                print("Re-run NDVI download to generate *_NDVI_raw.tif files.")
            elif result.get("cancelled"):
                print("\nCancelled.")
            elif "outputs" in result:
                print(f"\nHistorical average: {result['used']} files used "
                      f"(of {result['matched']} matched).")
                print(f"  → {ndvi_dir.resolve()}")
            else:
                print(f"\nNot enough matching files (matched={result.get('matched', 0)}, "
                      f"used={result.get('used', 0)}). Need at least 2.")
        except Exception as e:
            print(f"\nFATAL ERROR: {e}")
        finally:
            self.run_btn.configure(state="normal")
            self.hist_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")
            self.progress.stop()
            self.progress.configure(mode="indeterminate")
            self.progress_lbl.configure(text="")

    def _cancel(self):
        if self._cancel_event:
            self._cancel_event.set()
            self.cancel_btn.configure(state="disabled")
            self.progress_lbl.configure(text="Cancelling…")

    def _on_progress(self, done, total):
        self.after(0, lambda d=done, t=total: self._set_progress(d, t))

    def _set_progress(self, done, total):
        if total == 0:
            self.progress_lbl.configure(text="0 images")
            return
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=total, value=done)
        self.progress_lbl.configure(text=f"{done} / {total}")

    def _worker(self, shapefile, output, cloud, workers, do_rgb, do_ndwi, do_ndvi, do_fc, do_dem):
        try:
            import importlib
            import finder
            importlib.reload(finder)
            finder.run(shapefile, output,
                       cloud_max=cloud, workers=workers,
                       do_rgb=do_rgb, do_ndwi=do_ndwi, do_ndvi=do_ndvi,
                       do_false_color=do_fc, do_dem=do_dem,
                       cancel_event=self._cancel_event,
                       progress_callback=self._on_progress)
        except Exception as e:
            print(f"\nFATAL ERROR: {e}")
        finally:
            self.run_btn.configure(state="normal")
            self.hist_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")
            self.progress.stop()
            self.progress.configure(mode="indeterminate")
            self.progress_lbl.configure(text="")


if __name__ == "__main__":
    app = App()
    app.mainloop()
