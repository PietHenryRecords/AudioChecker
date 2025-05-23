import wave
import struct
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
import subprocess
import tempfile
import os

class AudioChecker:
    """
    Analysiert .wav- und .mp3-Dateien und erstellt einen PDF-Report.
    Für MP3-Support muss ffmpeg im PATH sein.
    """
    def __init__(self, filepath: str):
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in ('.wav', '.mp3'):
            raise ValueError("Nur .wav- und .mp3-Dateien werden unterstützt.")
        self.filepath = filepath
        self.params = None
        self.frames = None
        self.signal = None

    def load_wav(self):
        """
        Lädt eine WAV-Datei. Wenn die Eingabe eine MP3 ist,
        wird sie vorher per ffmpeg in eine temporäre WAV konvertiert.
        """
        filepath = self.filepath
        temp_wav = None

        if filepath.lower().endswith('.mp3'):
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav = tmp.name
            tmp.close()
            subprocess.run([
                'ffmpeg', '-y', '-i', filepath,
                '-ar', '44100', '-ac', '2', temp_wav
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            filepath = temp_wav

        with wave.open(filepath, 'rb') as wf:
            self.params = wf.getparams()  # (nchannels, sampwidth, framerate, nframes, comptype, compname)
            raw = wf.readframes(self.params.nframes)

        if temp_wav:
            try:
                os.remove(temp_wav)
            except OSError:
                pass

        # Unpacken in NumPy-Array
        fmt = '<' + 'h' * (self.params.nframes * self.params.nchannels)
        data = struct.unpack(fmt, raw)
        signal = np.array(data)
        if self.params.nchannels > 1:
            signal = signal.reshape(-1, self.params.nchannels)

        self.frames = self.params.nframes
        self.signal = signal

    def analyze(self):
        """
        Gibt ein Dict mit folgenden Werten zurück:
        channels, sample_width, framerate, frames,
        duration_s, peak_amplitude, mean_amplitude
        """
        if self.signal is None:
            raise RuntimeError("Audio nicht geladen. Bitte load_wav() vorher aufrufen.")

        duration = self.frames / self.params.framerate
        peak = np.max(np.abs(self.signal))
        mean_amp = np.mean(np.abs(self.signal))

        return {
            'channels': self.params.nchannels,
            'sample_width': self.params.sampwidth,
            'framerate': self.params.framerate,
            'frames': self.frames,
            'duration_s': duration,
            'peak_amplitude': int(peak),
            'mean_amplitude': float(mean_amp)
        }

    def plot_waveform(self):
        """
        Erstellt eine Wellenform-Grafik im PNG-Format und gibt
        einen BytesIO-Puffer zurück.
        """
        if self.signal is None:
            raise RuntimeError("Audio nicht geladen. Bitte load_wav() vorher aufrufen.")

        plt.figure()
        if self.params.nchannels > 1:
            plt.plot(self.signal[:, 0], label='Left')
            plt.plot(self.signal[:, 1], label='Right')
            plt.legend()
        else:
            plt.plot(self.signal)

        plt.title('Waveform')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')

        buf = BytesIO()
        plt.savefig(buf, format='PNG')
        plt.close()
        buf.seek(0)
        return buf

    def export_pdf_report(self, output_pdf: str):
        """
        Erstellt ein A4-PDF mit Analysewerten und Wellenform-Grafik.
        """
        analysis = self.analyze()
        waveform_img = self.plot_waveform()

        c = canvas.Canvas(output_pdf, pagesize=A4)
        width, height = A4

        c.setFont('Helvetica-Bold', 14)
        c.drawString(30, height - 50, 'AudioChecker Report')
        c.setFont('Helvetica', 12)

        y = height - 80
        for key, value in analysis.items():
            c.drawString(30, y, f"{key.replace('_', ' ').title()}: {value}")
            y -= 20

        c.drawImage(waveform_img, 30, y - 300, width=500, preserveAspectRatio=True)
        c.showPage()
        c.save()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='AudioChecker: Analysiere .wav und .mp3 und erzeuge PDF-Reports'
    )
    parser.add_argument('input', help='Pfad zur Eingabedatei (.wav oder .mp3)')
    parser.add_argument('-o', '--output', default='report.pdf', help='Pfad zur Ausgabedatei (PDF)')
    args = parser.parse_args()

    checker = AudioChecker(args.input)
    checker.load_wav()
    checker.export_pdf_report(args.output)
    print(f"PDF-Report gespeichert unter: {args.output}")
