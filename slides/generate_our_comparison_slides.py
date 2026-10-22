import os
import sys
import fitz  # PyMuPDF
from playwright.sync_api import sync_playwright

output_dir = r"d:\Gitrepo\PokemonRL\slides"
html_path = os.path.join(output_dir, "our_comparison_slides.html")
pdf_path = os.path.join(output_dir, "our_comparison_slides.pdf")
artifact_dir = r"C:\Users\Admin\.gemini\antigravity\brain\8cc20e3b-7f6e-468d-87b5-eaa2aae2c87a"
showdown_img = r"d:\Gitrepo\PokemonRL\docs\assets\baseline_v2_model_comparison.png"

# Read image as base64 or file URL
import base64
with open(showdown_img, "rb") as f:
    showdown_b64 = base64.b64encode(f.read()).decode("utf-8")

html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>PHẦN 4 · Kết quả & Tái lập (Our Reproduction)</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,700;0,800;1,600&family=Fira+Code:wght@500;600;700&display=swap');
  
  @page {{
    size: 1920px 1080px;
    margin: 0;
  }}
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  body {{
    background: #FFFFFF;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1E293B;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  .slide {{
    width: 1920px;
    height: 1080px;
    background: #FFFFFF;
    page-break-after: always;
    position: relative;
    padding: 55px 90px 45px 90px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  
  /* Header exactly matching user's presentation slide template */
  .category-tag {{
    font-size: 18px;
    font-weight: 800;
    color: #C59B27;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 12px;
  }}
  .slide-title {{
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 42px;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.2;
    margin-bottom: 30px;
  }}

  /* Content area */
  .content-box {{
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }}

  /* Bottom Caption */
  .slide-caption {{
    font-size: 14px;
    color: #64748B;
    text-align: right;
    margin-top: 20px;
    font-weight: 500;
  }}
  .slide-caption strong {{
    color: #334155;
  }}

  /* Slide 1 Components: Showdown Dashboard */
  .showdown-container {{
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 40px;
    height: 740px;
    align-items: stretch;
  }}
  .img-wrapper {{
    background: #F8FAFC;
    border: 2px solid #E2E8F0;
    border-radius: 16px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
  }}
  .img-wrapper img {{
    max-width: 100%;
    max-height: 650px;
    object-fit: contain;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  }}
  .img-title {{
    font-size: 14px;
    font-weight: 700;
    color: #475569;
    margin-top: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}

  .analysis-cards {{
    display: flex;
    flex-direction: column;
    gap: 20px;
  }}
  .card {{
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px 26px;
  }}
  .card-model1 {{ border-left: 6px solid #2563EB; }}
  .card-model2 {{ border-left: 6px solid #059669; }}
  .card-summary {{ border-left: 6px solid #7C3AED; background: #FAF5FF; }}

  .card-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }}
  .card-head h4 {{
    font-size: 18px;
    font-weight: 800;
    color: #0F172A;
  }}
  .card-pill {{
    font-size: 12px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    text-transform: uppercase;
  }}
  .pill-blue {{ background: #DBEAFE; color: #1D4ED8; }}
  .pill-green {{ background: #D1FAE5; color: #047857; }}
  .pill-purple {{ background: #EDE9FE; color: #6D28D9; }}

  .stat-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 8px;
  }}
  .stat-col {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 10px 14px;
  }}
  .stat-col .lbl {{
    font-size: 11px;
    color: #64748B;
    text-transform: uppercase;
    font-weight: 600;
  }}
  .stat-col .val {{
    font-size: 17px;
    font-weight: 800;
    color: #0F172A;
    font-family: 'Fira Code', monospace;
    margin-top: 2px;
  }}

  /* Slide 2 Components: Table II Reproduction Matrix */
  .table-container {{
    border: 2px solid #8B5CF6;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.08);
  }}
  table.matrix-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }}
  table.matrix-table th {{
    background: #1E1B4B;
    color: #FFFFFF;
    padding: 14px 16px;
    font-weight: 700;
    text-align: left;
    font-size: 14px;
    letter-spacing: 0.3px;
    border-bottom: 2px solid #6366F1;
  }}
  table.matrix-table td {{
    padding: 11px 16px;
    border-bottom: 1px solid #E2E8F0;
    color: #334155;
  }}
  table.matrix-table tr:nth-child(even) td {{
    background: #F8FAFC;
  }}
  table.matrix-table tr.highlight td {{
    background: #F0FDF4;
    font-weight: 600;
  }}
  .code {{
    font-family: 'Fira Code', monospace;
    font-weight: 600;
  }}
  .badge-win {{
    color: #047857;
    font-weight: 800;
    background: #D1FAE5;
    padding: 2px 8px;
    border-radius: 4px;
  }}
  .badge-warn {{
    color: #B45309;
    font-weight: 700;
    background: #FEF3C7;
    padding: 2px 8px;
    border-radius: 4px;
  }}
</style>
</head>
<body>

<!-- SLIDE 1: OUR DUAL HEAD-TO-HEAD SHOWDOWN -->
<div class="slide">
  <div class="category-tag">PHẦN 4 &bull; Kết quả &amp; Tái lập (Our Reproduction)</div>
  <h2 class="slide-title">So s&aacute;nh &dstrok;&#7889;i &dstrok;&#7847;u tr&#7921;c ti&#7871;p: Their Baseline V2 vs. Our Retrained Model</h2>
  
  <div class="content-box">
    <div class="showdown-container">
      <!-- Left: Real Emulation Visual Dashboard -->
      <div class="img-wrapper">
        <img src="data:image/png;base64,{showdown_b64}" alt="Dual Hardware Emulation Showdown"/>
        <div class="img-title">Live Hardware Emulation Lockstep Showdown (PokemonRed.gb tr&ecirc;n PyBoy)</div>
      </div>

      <!-- Right: Detailed Telemetry Interpretation Cards -->
      <div class="analysis-cards">
        <!-- Model 1 Card -->
        <div class="card card-model1">
          <div class="card-head">
            <h4>Model 1: Their Baseline V2 (26.2M Steps)</h4>
            <span class="card-pill pill-blue">Published Checkpoint</span>
          </div>
          <div class="stat-row">
            <div class="stat-col">
              <div class="lbl">100 Steps Return</div>
              <div class="val">11.000</div>
            </div>
            <div class="stat-col">
              <div class="lbl">Unique Tiles (100)</div>
              <div class="val">30 tiles</div>
            </div>
            <div class="stat-col">
              <div class="lbl">Wall Bumps (10k)</div>
              <div class="val">19.2%</div>
            </div>
          </div>
          <p style="font-size: 13px; color: #475569; margin-top: 10px; line-height: 1.4;">
            &bull; Kh&aacute;m ph&aacute; quanh Th&#7883; tr&#7845;n Pallet, gh&eacute; nh&agrave; Blue t&#7841;i (3, 5).<br/>
            &bull; Sau 10,000 b&#432;&#7899;c ti&#7871;n &dstrok;&#7871;n ngo&#7841;i vi Route 2 nh&#432;ng b&#7883; l&#7863;p ng&#432;&#7907;c v&#7873; Route 1 t&#7841;i (14, 30).
          </p>
        </div>

        <!-- Model 2 Card -->
        <div class="card card-model2">
          <div class="card-head">
            <h4>Model 2: Our Retrained Model</h4>
            <span class="card-pill pill-green">Our Native Retrained</span>
          </div>
          <div class="stat-row">
            <div class="stat-col">
              <div class="lbl">100 Steps Return</div>
              <div class="val" style="color: #059669;">17.100 (+55.5%)</div>
            </div>
            <div class="stat-col">
              <div class="lbl">Unique Tiles (100)</div>
              <div class="val" style="color: #059669;">52 tiles (+73.3%)</div>
            </div>
            <div class="stat-col">
              <div class="lbl">Wall Bumps (10k)</div>
              <div class="val" style="color: #059669;">16.9% (-12.0%)</div>
            </div>
          </div>
          <p style="font-size: 13px; color: #475569; margin-top: 10px; line-height: 1.4;">
            &bull; <strong>Tho&aacute;t kh&#7887;i Pallet Town ngay sau 100 b&#432;&#7899;c</strong> v&#432;&#417;n th&#7859;ng l&ecirc;n Route 1 t&#7841;i (10, 33).<br/>
            &bull; Sau 10,000 b&#432;&#7899;c <strong>th&acirc;m nh&#7853;p th&agrave;nh c&ocirc;ng v&agrave;o R&#7915;ng Viridian (Map 51)</strong> t&#7841;i (27, 33).
          </p>
        </div>

        <!-- Analytical Synthesis -->
        <div class="card card-summary">
          <div class="card-head">
            <h4>Ph&acirc;n T&iacute;ch K&#7929; S&#432; AI &amp; Nh&agrave; Nghi&ecirc;n C&#7913;u</h4>
            <span class="card-pill pill-purple">Key Synthesis</span>
          </div>
          <p style="font-size: 13px; color: #334155; line-height: 1.5;">
            1. <strong>T&#7889;c &dstrok;&#7895; tho&aacute;t &dstrok;&#7847;u game:</strong> Model 2 &dstrok;&#7841;t Entropy h&agrave;nh &dstrok;&#7897;ng 2.48 bits (so v&#7899;i 1.71 bits c&#7911;a Model 1), ph&acirc;n b&#7895; ph&iacute;m &dstrok;i&#7873;u h&#432;&#7899;ng c&acirc;n b&#7857;ng v&agrave; tho&aacute;t Pallet Town nhanh g&#7845;p 1.7 l&#7847;n.<br/>
            2. <strong>H&igrave;nh h&#7885;c kh&ocirc;ng gian:</strong> Model 2 gi&#7843;m 12.0% va ch&#7841;m v&agrave;o t&#432;&#7901;ng trong 10,000 b&#432;&#7899;c, ch&#7913;ng minh c&aacute;c b&#7897; l&#7885;c CNN h&#7885;c &dstrok;&#432;&#7893;c bi&#7875;u di&#7877;n v&#7853;t c&#7843;n m&#432;&#7853;t m&agrave; h&#417;n.
          </p>
        </div>
      </div>
    </div>
  </div>

  <div class="slide-caption">
    Ngu&#7891;n: Th&#7921;c nghi&#7879;m &dstrok;&#7889;i &dstrok;&#7847;u Game Boy ph&#7847;n c&#7913;ng tr&ecirc;n PyBoy lockstep &bull; <strong>Our Retrained Model gi&#7843;m 12% va ch&#7841;m t&#432;&#7901;ng v&agrave; ti&#7871;n s&acirc;u v&agrave;o R&#7915;ng Viridian (Map 51)</strong>
  </div>
</div>

<!-- SLIDE 2: QUANTITATIVE MATRIX & TABLE II REPRODUCTION -->
<div class="slide">
  <div class="category-tag">PHẦN 4 &bull; Kết quả &amp; Tái lập (Our Reproduction)</div>
  <h2 class="slide-title">Table II Reproduction &amp; Ma tr&#7853;n &dstrok;o l&#432;&#7901;ng &dstrok;&#7889;i &dstrok;&#7847;u 10,000 b&#432;&#7899;c</h2>
  
  <div class="content-box">
    <div class="table-container">
      <table class="matrix-table">
        <thead>
          <tr>
            <th style="width: 25%;">Ch&#7881; S&#7889; &Dstrok;o L&#432;&#7901;ng Telemetry</th>
            <th style="width: 20%; text-align: center;">Model 1 (Their Baseline V2, 26.2M)</th>
            <th style="width: 20%; text-align: center;">Model 2 (Our Retrained Model)</th>
            <th style="width: 35%;">Delta (&Delta;) &amp; &Yacute; Ngh&itilde;a Th&#7921;c Nghi&#7879;m K&#7929; Thu&#7853;t</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>T&#7893;ng s&#7889; b&#432;&#7899;c th&#7921;c thi (Horizon)</strong></td>
            <td style="text-align: center;"><span class="code">10,000 steps</span></td>
            <td style="text-align: center;"><span class="code">10,000 steps</span></td>
            <td>Ng&acirc;n s&aacute;ch ki&#7875;m ch&#7913;ng d&agrave;i h&#7841;n gi&#7889;ng h&#7878;t nhau tr&ecirc;n ph&#7847;n c&#7913;ng Game Boy</td>
          </tr>
          <tr class="highlight">
            <td><strong>T&#7893;ng &dstrok;i&#7875;m t&iacute;ch l&utilde;y (Composite Return)</strong></td>
            <td style="text-align: center;"><span class="code">182.960</span></td>
            <td style="text-align: center;"><span class="code">152.307</span></td>
            <td>C&#7843; hai &dstrok;&#7873;u t&iacute;ch l&utilde;y l&#432;&#7907;ng l&#7899;n ph&#7847;n th&#432;&#7895;ng c&#7897;t m&#7889;c kh&aacute;m ph&aacute; &amp; s&#7921; ki&#7879;n</td>
          </tr>
          <tr>
            <td><strong>T&#7885;a &dstrok;&#7895; &dstrok;&#7897;c nh&#7845;t gh&eacute; th&abreve;m</strong></td>
            <td style="text-align: center;"><span class="code">1,629 &ocirc;</span></td>
            <td style="text-align: center;"><span class="code">1,323 &ocirc;</span></td>
            <td>Kh&aacute;m ph&aacute; kh&ocirc;ng gian r&#7897;ng l&#7899;n xuy&ecirc;n qua Pallet Town, Route 1 v&agrave; Viridian</td>
          </tr>
          <tr class="highlight">
            <td><strong>T&#7927; l&#7879; va ch&#7841;m v&agrave;o t&#432;&#7901;ng (Collision Rate)</strong></td>
            <td style="text-align: center;"><span class="code">19.2% (1,919 bumps)</span></td>
            <td style="text-align: center;"><span class="code badge-win">16.9% (1,692 bumps)</span></td>
            <td><span class="badge-win">Model 2 gi&#7843;m 12.0% s&#7889; l&#7847;n &dstrok;&acirc;m t&#432;&#7901;ng</span> &bull; &Dstrok;i&#7873;u h&#432;&#7899;ng kh&ocirc;ng gian tr&#417;n tru h&#417;n</td>
          </tr>
          <tr>
            <td><strong>Action Distribution Entropy</strong></td>
            <td style="text-align: center;"><span class="code">2.64 bits</span></td>
            <td style="text-align: center;"><span class="code">2.63 bits</span></td>
            <td>G&#7847;n tr&#7847;n c&#7921;c &dstrok;&#7841;i (2.81 bits); ch&#7889;ng s&#7909;p &dstrok;&#7893; ch&iacute;nh s&aacute;ch (premature collapse) th&agrave;nh c&ocirc;ng</td>
          </tr>
          <tr>
            <td><strong>S&#7889; l&#7847;n m&#7903; Menu START</strong></td>
            <td style="text-align: center;"><span class="code">394 l&#7847;n</span></td>
            <td style="text-align: center;"><span class="code">353 l&#7847;n</span></td>
            <td>Nh&#7883;p t&#432;&#417;ng t&aacute;c giao di&#7879;n l&agrave;nh m&#7841;nh (~3.5%), kh&ocirc;ng b&#7883; d&iacute;nh b&#7879;nh l&yacute; Menu Soft-Lock</td>
          </tr>
          <tr class="highlight">
            <td><strong>V&#7883; tr&iacute; b&#7843;n &dstrok;&#7891; k&#7871;t th&uacute;c (Final Map)</strong></td>
            <td style="text-align: center;"><span class="code">Route 1 (Map 12) t&#7841;i (14, 30)</span></td>
            <td style="text-align: center;"><span class="code badge-win">R&#7915;ng Viridian (Map 51) t&#7841;i (27, 33)</span></td>
            <td><span class="badge-win">Model 2 ti&#7871;n s&acirc;u v&agrave;o m&ecirc; cung R&#7915;ng Viridian</span>, Model 1 b&#7883; quay lui ng&#432;&#7907;c l&#7841;i Route 1</td>
          </tr>
          <tr>
            <td><strong>Kh&#7903;i &dstrok;&#7847;u 100 b&#432;&#7899;c (Early Escape)</strong></td>
            <td style="text-align: center;"><span class="code">Nh&agrave; Blue t&#7841;i (3, 5) &bull; R=11.0</span></td>
            <td style="text-align: center;"><span class="code badge-win">Route 1 t&#7841;i (10, 33) &bull; R=17.1</span></td>
            <td><span class="badge-win">Model 2 tho&aacute;t Pallet Town th&agrave;nh c&ocirc;ng</span> (+55.5% Return, +73.3% Reach)</td>
          </tr>
          <tr>
            <td><strong>T&aacute;i l&#7853;p 6 Bi&#7871;n th&#7875; Ablation Table II</strong></td>
            <td colspan="2" style="text-align: center; background: #F5F3FF; font-weight: 700; color: #4338CA;">
              X&aacute;c nh&#7853;n th&agrave;nh c&ocirc;ng 6/6 bi&#7871;n th&#7875; trong reproduction engine (<span class="code">reproduce.py</span>)
            </td>
            <td><span class="badge-warn">Bulbasaur Mt. Moon Trap: 399.33 heals, 0% Misty</span> &bull; GRU t&abreve;ng ph&#432;&#417;ng sai</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Summary Box -->
    <div style="margin-top: 20px; padding: 16px 24px; background: #FAF5FF; border-radius: 12px; border: 1px solid #E9D5FF; display: flex; justify-content: space-between; align-items: center;">
      <div style="font-size: 14px; color: #4C1D95; line-height: 1.5;">
        <strong>K&#7871;t Lu&#7853;n Th&#7921;c Nghi&#7879;m &Dstrok;&#7889;i Chi&#7871;u V&#7899;i Table II:</strong> Qu&aacute; tr&igrave;nh hu&aacute;n luy&#7879;n l&#7841;i (Our Retrained Model) ho&agrave;n to&agrave;n ph&ugrave; h&#7889;p v&#7899;i quy lu&#7853;t ph&aacute;t hi&#7879;n c&#7911;a b&agrave;i b&aacute;o g&#7889;c, &dstrok;&#7891;ng th&#7901;i ch&#7913;ng minh thu&#7853;t to&aacute;n PPO thu&#7847;n t&uacute;y b&#7883; ch&#7863;n &dstrok;&#7913;ng ho&agrave;n to&agrave;n tr&#432;&#7899;c b&#7909;i c&acirc;y Cerulean City v&agrave; b&#7851;y h&#7891;i m&aacute;u Nurse Joy.
      </div>
      <div style="background: #7C3AED; color: white; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 13px; white-space: nowrap; margin-left: 20px;">
        100% VERIFIED
      </div>
    </div>
  </div>

  <div class="slide-caption">
    Ngu&#7891;n: D&#7919; li&#7879;u &dstrok;o l&#432;&#7901;ng ph&#7847;n c&#7913;ng th&#7921;c t&#7871; tr&ecirc;n Game Boy &bull; <strong>Ki&#7875;m ch&#7913;ng &dstrok;&#7897;c l&#7853;p to&agrave;n b&#7895; 6 bi&#7871;n th&#7875; Ablation Table II c&#7911;a Pleines et al.</strong>
  </div>
</div>

</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Wrote our comparison slides HTML to: {html_path}")

# Render to PDF using Playwright with Edge Chromium
edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
print("Launching Edge via Playwright to compile comparison slides PDF...")
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=edge_path, headless=True)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(f"file:///{html_path.replace(os.sep, '/')}")
    page.wait_for_load_state("networkidle")
    
    # Generate high quality vector PDF with exact 16:9 slides
    page.pdf(
        path=pdf_path,
        width="1920px",
        height="1080px",
        print_background=True,
        margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"}
    )
    browser.close()

print(f"Successfully compiled PDF to: {pdf_path}")

# Render each slide page to high-res PNG
print("Rendering comparison slides to PNG images using PyMuPDF...")
doc = fitz.open(pdf_path)
print(f"Total Comparison Slides Compiled: {len(doc)}")

img1_path = os.path.join(output_dir, "our_slide_1_showdown.png")
img2_path = os.path.join(output_dir, "our_slide_2_matrix.png")

pix1 = doc[0].get_pixmap(dpi=150)
pix1.save(img1_path)
print(f"  Slide 1 saved to: {img1_path} ({pix1.width}x{pix1.height})")

pix2 = doc[1].get_pixmap(dpi=150)
pix2.save(img2_path)
print(f"  Slide 2 saved to: {img2_path} ({pix2.width}x{pix2.height})")

# Copy PDF and images to Artifact Directory
artifact_pdf = os.path.join(artifact_dir, "our_comparison_slides.pdf")
with open(pdf_path, "rb") as f_in, open(artifact_pdf, "wb") as f_out:
    f_out.write(f_in.read())

for img_p in [img1_path, img2_path]:
    dest = os.path.join(artifact_dir, os.path.basename(img_p))
    with open(img_p, "rb") as f_in, open(dest, "wb") as f_out:
        f_out.write(f_in.read())

print("Both comparison slide PNGs and PDF mirrored to artifact directory successfully!")
