"""
energy_deposition.py
--------------------
Analise da deposicao de energia nos detetores.

Analises incluidas:
    1. Histograma de deposicao de energia em cada detetor.
    2. Histograma de deposicao de energia por tipo de particula
       (muoes, pioes e outras) em cada detetor.
    3. Histograma da perda de energia total nos detetores
       para cada tipo de particula.
"""

import os
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("Plain")
ROOT.gStyle.SetOptStat(1)
ROOT.gStyle.SetOptTitle(1)

DATA_FILE = "data/AmberTarget_Run_0.root"
OUTPUT_DIR = "plots/energy"

# Fator de conversao eV -> MeV
EV_TO_MEV = 1e-6


def plot_energy_deposition_per_detector():
    """
    Histograma de deposicao de energia em cada detetor.
    Tree: edep_Per_Event, campos detector0..3 (em eV) — convertidos para MeV.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ficheiro = ROOT.TFile(DATA_FILE, "READ")
    dados = ficheiro.Get("edep_Per_Event")

    nBins = 200
    minBin = 0.0
    maxBin = 0.6  # 0.6 MeV — cobre toda a distribuicao

    for i in range(4):
        canvas = ROOT.TCanvas("c_edep_{}".format(i), "", 800, 600)
        canvas.SetLogy()

        histoName = "hEdepDet{}".format(i)
        branchName = "detector{}".format(i)

        histo = ROOT.TH1D(histoName,
                          "Deposicao de energia - Detetor {};Energia (MeV);Contagens".format(i),
                          nBins, minBin, maxBin)

        # converter eV para MeV na expressao
        dados.Draw("{}*{}>>{}" .format(branchName, EV_TO_MEV, histoName),
                   "{}>0".format(branchName), "goff")

        histo.SetFillColor(ROOT.kBlue - 9)
        histo.SetLineColor(ROOT.kBlue + 1)
        histo.Draw("HIST")

        canvas.SaveAs("{}/edep_detector{}.png".format(OUTPUT_DIR, i))
        canvas.Close()

    ficheiro.Close()
    print("[OK] plot_energy_deposition_per_detector")


def plot_energy_per_particle_per_detector():
    """
    Histograma de deposicao de energia por tipo de particula em cada detetor.
    Tree: tracksData, campos EdepDet0_keV..EdepDet3_keV — convertidos para MeV.
    Particulas: muoes (PDG +/-13), pioes (PDG +/-211), outras (e-, e+, p, kaoes, etc.).
    Histogramas normalizados a 1 para comparar formas independentemente da abundancia.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ficheiro = ROOT.TFile(DATA_FILE, "READ")
    dados = ficheiro.Get("tracksData")

    # fator de conversao keV -> MeV
    KEV_TO_MEV = 1e-3

    nBins = 200
    minBin = 0.0
    maxBin = 10.0  # 10 MeV — cobre a regiao com dados significativos

    particulas = [
        ("Muoes",  ROOT.kBlue,    "(particlePDG==13 || particlePDG==-13)"),
        ("Pioes",  ROOT.kRed,     "(particlePDG==211 || particlePDG==-211)"),
        ("Outras (e, p, K, ...)", ROOT.kGreen+2, "(particlePDG!=13 && particlePDG!=-13 && particlePDG!=211 && particlePDG!=-211)"),
    ]

    for i in range(4):
        branchName = "EdepDet{}_keV".format(i)
        exprMeV = "{}*{}".format(branchName, KEV_TO_MEV)

        canvas = ROOT.TCanvas("c_part_det{}".format(i), "", 800, 600)
        canvas.SetLogy()

        legenda = ROOT.TLegend(0.58, 0.70, 0.88, 0.88)
        legenda.SetBorderSize(1)

        histos = []
        for j, (nome, cor, selecao) in enumerate(particulas):
            histoName = "hPart{}_det{}".format(j, i)
            selecao_completa = "{} && {}>0".format(selecao, branchName)

            histo = ROOT.TH1D(histoName,
                              "Deposicao de energia por particula - Detetor {};Energia (MeV);Densidade de probabilidade".format(i),
                              nBins, minBin, maxBin)

            dados.Draw("{}>>{}".format(exprMeV, histoName),
                       selecao_completa, "goff")

            # normalizar a 1 para comparar formas
            if histo.Integral() > 0:
                histo.Scale(1.0 / histo.Integral())

            histo.SetLineColor(cor)
            histo.SetLineWidth(2)

            opcao = "HIST" if j == 0 else "HIST SAME"
            histo.Draw(opcao)
            legenda.AddEntry(histo, nome, "l")
            histos.append(histo)

        # ajustar eixo Y
        maximo = max(h.GetMaximum() for h in histos)
        histos[0].SetMaximum(maximo * 5)
        histos[0].SetMinimum(1e-6)

        ROOT.gStyle.SetOptStat(0)
        legenda.Draw()
        canvas.SaveAs("{}/edep_por_particula_detector{}.png".format(OUTPUT_DIR, i))
        canvas.Close()

    ROOT.gStyle.SetOptStat(1)
    ficheiro.Close()
    print("[OK] plot_energy_per_particle_per_detector")


def plot_total_energy_loss_per_particle():
    """
    Histograma da perda de energia total (soma dos 4 detetores) por tipo de particula.
    Tree: tracksData. Valores em MeV. Normalizado a 1 para comparar formas.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ficheiro = ROOT.TFile(DATA_FILE, "READ")
    dados = ficheiro.Get("tracksData")

    KEV_TO_MEV = 1e-3

    nBins = 200
    minBin = 0.0
    maxBin = 30.0  # 30 MeV — cobre a regiao com dados significativos

    somaExpr = "(EdepDet0_keV+EdepDet1_keV+EdepDet2_keV+EdepDet3_keV)*{}".format(KEV_TO_MEV)
    somaOriginal = "EdepDet0_keV+EdepDet1_keV+EdepDet2_keV+EdepDet3_keV"

    particulas = [
        ("Muoes",  ROOT.kBlue,    "(particlePDG==13 || particlePDG==-13)"),
        ("Pioes",  ROOT.kRed,     "(particlePDG==211 || particlePDG==-211)"),
        ("Outras (e, p, K, ...)", ROOT.kGreen+2, "(particlePDG!=13 && particlePDG!=-13 && particlePDG!=211 && particlePDG!=-211)"),
    ]

    canvas = ROOT.TCanvas("c_total", "", 800, 600)
    canvas.SetLogy()

    legenda = ROOT.TLegend(0.58, 0.70, 0.88, 0.88)
    legenda.SetBorderSize(1)

    histos = []
    for j, (nome, cor, selecao) in enumerate(particulas):
        histoName = "hTotal{}".format(j)
        selecao_completa = "{} && ({})>0".format(selecao, somaOriginal)

        histo = ROOT.TH1D(histoName,
                          "Perda de energia total por particula;Energia total (MeV);Densidade de probabilidade",
                          nBins, minBin, maxBin)

        dados.Draw("{}>>{}".format(somaExpr, histoName),
                   selecao_completa, "goff")

        # normalizar a 1 para comparar formas
        if histo.Integral() > 0:
            histo.Scale(1.0 / histo.Integral())

        histo.SetLineColor(cor)
        histo.SetLineWidth(2)

        opcao = "HIST" if j == 0 else "HIST SAME"
        histo.Draw(opcao)
        legenda.AddEntry(histo, nome, "l")
        histos.append(histo)

    maximo = max(h.GetMaximum() for h in histos)
    histos[0].SetMaximum(maximo * 5)
    histos[0].SetMinimum(1e-8)

    ROOT.gStyle.SetOptStat(0)
    legenda.Draw()
    canvas.SaveAs("{}/edep_total_por_particula.png".format(OUTPUT_DIR))
    canvas.Close()

    ficheiro.Close()
    print("[OK] plot_total_energy_loss_per_particle")


def plot_landau_fit_per_detector():
    """
    Ajuste de uma distribuicao de Landau a deposicao de energia de pioes em cada detetor.

    Filtros aplicados (igual ao macro fitLandau.C):
        - particlePDG == +/-211  (pioes carregados)
        - EdepDet > 10 keV       (remover deposicoes residuais / ruido)
        - momentum_GeV > 0.5     (pioes relativistas — regiao de minimo de ionizacao)

    Parametros extraidos do fit:
        - MPV   (Most Probable Value) — energia de deposicao mais provavel
        - Sigma — largura da distribuicao de Landau
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    KEV_TO_MEV = 1e-3

    ficheiro = ROOT.TFile(DATA_FILE, "READ")
    dados = ficheiro.Get("tracksData")

    nBins = 300
    minBin = 0.0
    maxBin = 8.0  # MeV — cobre o pico de Landau dos pioes

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(1)  # mostrar parametros do fit no canvas

    for i in range(4):
        branchName = "EdepDet{}_keV".format(i)
        exprMeV = "{}*{}".format(branchName, KEV_TO_MEV)

        # pioes relativistas com deposicao significativa (igual ao macro)
        selecao = "(particlePDG==211 || particlePDG==-211) && {}>10 && momentum_GeV>0.5".format(branchName)

        canvas = ROOT.TCanvas("c_landau_{}".format(i), "", 800, 600)

        histoName = "hLandau_det{}".format(i)
        histo = ROOT.TH1D(histoName,
                          "Deposicao de energia de pioes - Detetor {} (fit Landau);Energia (MeV);Contagens".format(i),
                          nBins, minBin, maxBin)

        dados.Draw("{}>>{}".format(exprMeV, histoName), selecao, "goff")

        histo.SetLineColor(ROOT.kRed - 3)
        histo.SetFillColor(ROOT.kRed - 9)
        histo.SetLineWidth(1)

        # fit Landau na regiao do pico (evitar a cauda que distorce o fit)
        landauFit = ROOT.TF1("landauFit_{}".format(i), "landau", 0.3, 5.0)
        landauFit.SetLineColor(ROOT.kBlack)
        landauFit.SetLineWidth(2)

        histo.Fit(landauFit, "RQ")  # R = range da funcao, Q = silencioso

        mpv       = landauFit.GetParameter(1)
        sigma     = landauFit.GetParameter(2)
        mpv_err   = landauFit.GetParError(1)
        sigma_err = landauFit.GetParError(2)

        histo.Draw("HIST")
        landauFit.Draw("SAME")

        canvas.SaveAs("{}/landau_pioes_detector{}.png".format(OUTPUT_DIR, i))
        canvas.Close()

        print("  Detetor {}: MPV = {:.3f} +/- {:.3f} MeV | Sigma = {:.3f} +/- {:.3f} MeV".format(
            i, mpv, mpv_err, sigma, sigma_err))

    ROOT.gStyle.SetOptStat(1)
    ROOT.gStyle.SetOptFit(0)
    ficheiro.Close()
    print("[OK] plot_landau_fit_per_detector")


def plot_mpv_vs_detector():
    """
    Grafico de MPV (Most Probable Value) do fit Landau em funcao do numero do detetor.
    Mostra a perda de energia progressiva dos pioes ao longo do trajeto.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    KEV_TO_MEV = 1e-3

    ficheiro = ROOT.TFile(DATA_FILE, "READ")
    dados = ficheiro.Get("tracksData")

    nBins = 300
    minBin = 0.0
    maxBin = 8.0

    mpv_values   = []
    mpv_errors   = []
    sigma_values = []
    sigma_errors = []

    for i in range(4):
        branchName = "EdepDet{}_keV".format(i)
        exprMeV = "{}*{}".format(branchName, KEV_TO_MEV)
        selecao = "(particlePDG==211 || particlePDG==-211) && {}>10 && momentum_GeV>0.5".format(branchName)

        histoName = "hMPV_det{}".format(i)
        histo = ROOT.TH1D(histoName, "", nBins, minBin, maxBin)
        dados.Draw("{}>>{}".format(exprMeV, histoName), selecao, "goff")

        landauFit = ROOT.TF1("landauMPV_{}".format(i), "landau", 0.3, 5.0)
        histo.Fit(landauFit, "RQ")

        mpv_values.append(landauFit.GetParameter(1))
        mpv_errors.append(landauFit.GetParError(1))
        sigma_values.append(landauFit.GetParameter(2))
        sigma_errors.append(landauFit.GetParError(2))

    ficheiro.Close()

    # construir o grafico com barras de erro
    n = 4
    detectors = ROOT.TGraphErrors(n)

    for i in range(n):
        detectors.SetPoint(i, i, mpv_values[i])
        detectors.SetPointError(i, 0.0, mpv_errors[i])

    canvas = ROOT.TCanvas("c_mpv", "", 800, 600)
    canvas.SetGrid()

    detectors.SetTitle("MPV do fit Landau por detetor (pioes, p > 0.5 GeV/c);Numero do detetor;MPV (MeV)")
    detectors.SetMarkerStyle(21)
    detectors.SetMarkerSize(1.5)
    detectors.SetMarkerColor(ROOT.kRed + 1)
    detectors.SetLineColor(ROOT.kRed + 1)
    detectors.SetLineWidth(2)
    detectors.Draw("AP")

    # ajustar eixo X para mostrar so 0,1,2,3
    detectors.GetXaxis().SetLimits(-0.5, 3.5)
    detectors.GetXaxis().SetNdivisions(4)
    detectors.GetYaxis().SetTitleOffset(1.3)

    canvas.SaveAs("{}/mpv_vs_detector.png".format(OUTPUT_DIR))
    canvas.Close()

    print("[OK] plot_mpv_vs_detector")
    for i in range(n):
        print("  Detetor {}: MPV = {:.4f} +/- {:.4f} MeV".format(i, mpv_values[i], mpv_errors[i]))


if __name__ == "__main__":
    plot_energy_deposition_per_detector()
    plot_energy_per_particle_per_detector()
    plot_total_energy_loss_per_particle()
    plot_landau_fit_per_detector()
    plot_mpv_vs_detector()
