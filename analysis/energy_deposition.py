"""
energy_deposition.py
--------------------
Analise da deposicao de energia nos detetores.

Analises incluidas:
    1.  Composicao do feixe — abundancia relativa de cada especie de particula.
    2.  Histograma de deposicao de energia em cada detetor.
    3.  Histograma de deposicao de energia por tipo de particula em cada detetor.
    4.  Histograma da perda de energia total nos detetores por tipo de particula.
    5.  Fit de distribuicao de Landau a pioes e protoes por detetor.
    6.  MPV e sigma do fit Landau vs numero de detetor (com ajuste linear).
    7.  Fit de Landau para kaoes — comparacao da serie piao/kaon/protao.
    8.  MPV vs massa da particula — confirmacao da formula de Bethe-Bloch.
    9.  Sobreposicao dos 4 detetores para verificacao de consistencia.
    10. dE/dx vs momento — bandas de Bethe-Bloch para identificacao de particulas.
    11. Resolucao relativa (sigma/MPV) e poder de separacao entre especies.
"""

import os
import math
import array as arr
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("Plain")
ROOT.gStyle.SetOptStat(1)
ROOT.gStyle.SetOptTitle(1)

DATA_FILES = [
    "data/AmberTarget_Run_0.root",
    "data/AmberTarget_Run_1.root",
    "data/AmberTarget_Run_2.root",
    "data/AmberTarget_Run_3.root",
]
OUTPUT_DIR = "plots/energy"

# Fator de conversao eV -> MeV
EV_TO_MEV = 1e-6


def carregar_chain(tree_name):
    """Carrega uma TTree de todos os runs numa TChain."""
    chain = ROOT.TChain(tree_name)
    for f in DATA_FILES:
        chain.Add(f)
    return chain


def plot_beam_composition():
    """
    Composicao do feixe — numero de tracks por especie de particula.
    Identifica as especies dominantes nos dados e a sua abundancia relativa.
    Essencial para contextualizar todas as analises subsequentes.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dados = carregar_chain("tracksData")

    # especies identificadas nos dados por PDG code
    especies = [
        ("e^{#pm}",    "(particlePDG==11  || particlePDG==-11)",  ROOT.kOrange+1),
        ("#gamma",     "particlePDG==22",                          ROOT.kGreen+2),
        ("#pi^{#pm}",  "(particlePDG==211 || particlePDG==-211)",  ROOT.kRed+1),
        ("p",          "particlePDG==2212",                        ROOT.kMagenta+1),
        ("K^{#pm}",    "(particlePDG==321 || particlePDG==-321)",  ROOT.kCyan+1),
        ("#mu^{#pm}",  "(particlePDG==13  || particlePDG==-13)",   ROOT.kBlue+1),
    ]

    nomes  = []
    counts = []
    cores  = []

    for nome, selecao, cor in especies:
        h = ROOT.TH1D("hCount_{}".format(nome), "", 1, 0, 1)
        dados.Draw("0.5>>hCount_{}".format(nome), selecao, "goff")
        nomes.append(nome)
        counts.append(h.GetEntries())
        cores.append(cor)

    canvas = ROOT.TCanvas("c_beam", "", 900, 600)
    canvas.SetLogy()
    canvas.SetGrid()
    canvas.SetBottomMargin(0.12)

    n = len(nomes)
    hbar = ROOT.TH1D("hBeam",
                     "Composicao do feixe (4 runs);Especie de particula;Numero de tracks",
                     n, 0, n)

    for i, (nome, count, cor) in enumerate(zip(nomes, counts, cores)):
        hbar.GetXaxis().SetBinLabel(i + 1, nome)
        hbar.SetBinContent(i + 1, count)

    hbar.SetFillColor(ROOT.kBlue - 9)
    hbar.SetLineColor(ROOT.kBlue + 1)
    hbar.GetXaxis().SetLabelSize(0.06)
    hbar.GetYaxis().SetTitleOffset(1.3)

    ROOT.gStyle.SetOptStat(0)
    hbar.Draw("HIST")

    # adicionar percentagens abaixo do topo de cada barra
    total = sum(counts)
    labels = []  # manter referencias para evitar garbage collection
    for i, count in enumerate(counts):
        pct = 100.0 * count / total
        # posicionar o texto a 20% da altura da barra (em log scale)
        ypos = count * 0.3
        label = ROOT.TLatex(i + 0.5, ypos, "{:.1f}%".format(pct))
        label.SetTextSize(0.038)
        label.SetTextAlign(21)
        label.SetTextColor(ROOT.kBlack)
        label.Draw()
        labels.append(label)

    canvas.SaveAs("{}/composicao_feixe.png".format(OUTPUT_DIR))
    canvas.Close()

    ROOT.gStyle.SetOptStat(1)
    print("[OK] plot_beam_composition")
    print("  Total de tracks: {:,}".format(int(total)))
    for nome, count in zip(nomes, counts):
        print("  {:12s}: {:>9,.0f}  ({:.1f}%)".format(
            nome, count, 100.0 * count / total))


def plot_energy_deposition_per_detector():
    """
    Histograma de deposicao de energia em cada detetor.
    Tree: edep_Per_Event, campos detector0..3 (em eV) — convertidos para MeV.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dados = carregar_chain("edep_Per_Event")

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

    print("[OK] plot_energy_deposition_per_detector")


def plot_energy_per_particle_per_detector():
    """
    Histograma de deposicao de energia por tipo de particula em cada detetor.
    Tree: tracksData, campos EdepDet0_keV..EdepDet3_keV — convertidos para MeV.
    Particulas identificadas pelos PDG codes presentes nos dados:
        muoes (PDG +/-13), pioes (PDG +/-211), electroes/positroes (PDG +/-11),
        fotoes (PDG 22), protoes (PDG 2212), kaoes (PDG +/-321).
    Histogramas normalizados a 1 para comparar formas independentemente da abundancia.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dados = carregar_chain("tracksData")

    # fator de conversao keV -> MeV
    KEV_TO_MEV = 1e-3

    nBins = 200
    minBin = 0.0
    maxBin = 10.0  # 10 MeV — cobre a regiao com dados significativos

    particulas = [
        ("Muoes",              ROOT.kBlue,      "(particlePDG==13 || particlePDG==-13)"),
        ("Pioes",              ROOT.kRed,       "(particlePDG==211 || particlePDG==-211)"),
        ("Electroes/Positroes",ROOT.kOrange+1,  "(particlePDG==11 || particlePDG==-11)"),
        ("Fotoes",             ROOT.kGreen+2,   "(particlePDG==22)"),
        ("Protoes",            ROOT.kMagenta+1, "(particlePDG==2212)"),
        ("Kaoes",              ROOT.kCyan+1,    "(particlePDG==321 || particlePDG==-321)"),
    ]

    for i in range(4):
        branchName = "EdepDet{}_keV".format(i)
        exprMeV = "{}*{}".format(branchName, KEV_TO_MEV)

        canvas = ROOT.TCanvas("c_part_det{}".format(i), "", 800, 600)
        canvas.SetLogy()

        legenda = ROOT.TLegend(0.58, 0.58, 0.88, 0.88)
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
    print("[OK] plot_energy_per_particle_per_detector")


def plot_total_energy_loss_per_particle():
    """
    Histograma da perda de energia total (soma dos 4 detetores) por tipo de particula.
    Tree: tracksData. Valores em MeV. Normalizado a 1 para comparar formas.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dados = carregar_chain("tracksData")

    KEV_TO_MEV = 1e-3

    nBins = 200
    minBin = 0.0
    maxBin = 30.0  # 30 MeV — cobre a regiao com dados significativos

    somaExpr = "(EdepDet0_keV+EdepDet1_keV+EdepDet2_keV+EdepDet3_keV)*{}".format(KEV_TO_MEV)
    somaOriginal = "EdepDet0_keV+EdepDet1_keV+EdepDet2_keV+EdepDet3_keV"

    particulas = [
        ("Muoes",              ROOT.kBlue,      "(particlePDG==13 || particlePDG==-13)"),
        ("Pioes",              ROOT.kRed,       "(particlePDG==211 || particlePDG==-211)"),
        ("Electroes/Positroes",ROOT.kOrange+1,  "(particlePDG==11 || particlePDG==-11)"),
        ("Fotoes",             ROOT.kGreen+2,   "(particlePDG==22)"),
        ("Protoes",            ROOT.kMagenta+1, "(particlePDG==2212)"),
        ("Kaoes",              ROOT.kCyan+1,    "(particlePDG==321 || particlePDG==-321)"),
    ]

    canvas = ROOT.TCanvas("c_total", "", 800, 600)
    canvas.SetLogy()

    legenda = ROOT.TLegend(0.58, 0.58, 0.88, 0.88)
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

    dados = carregar_chain("tracksData")

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
    print("[OK] plot_landau_fit_per_detector")


def plot_mpv_vs_detector():
    """
    Grafico de MPV (Most Probable Value) do fit Landau em funcao do numero do detetor.
    Mostra a perda de energia progressiva dos pioes ao longo do trajeto.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    KEV_TO_MEV = 1e-3

    dados = carregar_chain("tracksData")

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

    # ajuste linear — o declive indica a perda de MPV por detetor
    ajusteLinear = ROOT.TF1("ajusteLinear", "pol1", -0.5, 3.5)
    ajusteLinear.SetLineColor(ROOT.kBlue + 1)
    ajusteLinear.SetLineWidth(2)
    ajusteLinear.SetLineStyle(2)  # tracejado para distinguir dos pontos
    detectors.Fit(ajusteLinear, "Q")

    declive     = ajusteLinear.GetParameter(1)
    declive_err = ajusteLinear.GetParError(1)

    ROOT.gStyle.SetOptFit(1)
    detectors.Draw("AP")
    ajusteLinear.Draw("SAME")

    canvas.SaveAs("{}/mpv_vs_detector.png".format(OUTPUT_DIR))
    canvas.Close()

    # --- grafico de sigma vs detetor ---
    grafSigma = ROOT.TGraphErrors(n)
    for i in range(n):
        grafSigma.SetPoint(i, i, sigma_values[i])
        grafSigma.SetPointError(i, 0.0, sigma_errors[i])

    canvasSigma = ROOT.TCanvas("c_sigma", "", 800, 600)
    canvasSigma.SetGrid()

    grafSigma.SetTitle("Sigma do fit Landau por detetor (pioes, p > 0.5 GeV/c);Numero do detetor;Sigma (MeV)")
    grafSigma.SetMarkerStyle(21)
    grafSigma.SetMarkerSize(1.5)
    grafSigma.SetMarkerColor(ROOT.kGreen + 2)
    grafSigma.SetLineColor(ROOT.kGreen + 2)
    grafSigma.SetLineWidth(2)

    ajusteSigma = ROOT.TF1("ajusteSigma", "pol1", -0.5, 3.5)
    ajusteSigma.SetLineColor(ROOT.kBlue + 1)
    ajusteSigma.SetLineWidth(2)
    ajusteSigma.SetLineStyle(2)
    grafSigma.Fit(ajusteSigma, "Q")

    ROOT.gStyle.SetOptFit(1)
    grafSigma.Draw("AP")
    ajusteSigma.Draw("SAME")

    grafSigma.GetXaxis().SetLimits(-0.5, 3.5)
    grafSigma.GetXaxis().SetNdivisions(4)
    grafSigma.GetYaxis().SetTitleOffset(1.3)

    canvasSigma.SaveAs("{}/sigma_vs_detector.png".format(OUTPUT_DIR))
    canvasSigma.Close()

    ROOT.gStyle.SetOptFit(0)
    print("[OK] plot_mpv_vs_detector")
    for i in range(n):
        print("  Detetor {}: MPV = {:.4f} +/- {:.4f} MeV | Sigma = {:.4f} +/- {:.4f} MeV".format(
            i, mpv_values[i], mpv_errors[i], sigma_values[i], sigma_errors[i]))
    print("  Declive MPV:   {:.4f} +/- {:.4f} MeV/detetor".format(declive, declive_err))
    print("  Declive Sigma: {:.4f} +/- {:.4f} MeV/detetor".format(
        ajusteSigma.GetParameter(1), ajusteSigma.GetParError(1)))


def plot_landau_fit_protons():
    """
    Ajuste de uma distribuicao de Landau a deposicao de energia de protoes em cada detetor.
    Filtros: momentum_GeV > 0.5 (protoes relativistas) e EdepDet > 10 keV.
    Comparacao com o MPV dos pioes mostra a diferenca de perda de energia
    entre particulas de massas distintas (Bethe-Bloch: dE/dx depende de massa e carga).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    KEV_TO_MEV = 1e-3

    dados = carregar_chain("tracksData")

    nBins = 300
    minBin = 0.0
    maxBin = 20.0  # MeV — protoes depositam muito mais energia que pioes

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(1)

    for i in range(4):
        branchName = "EdepDet{}_keV".format(i)
        exprMeV    = "{}*{}".format(branchName, KEV_TO_MEV)
        selecao    = "particlePDG==2212 && {}>10 && momentum_GeV>0.5".format(branchName)

        canvas   = ROOT.TCanvas("c_landau_p_{}".format(i), "", 800, 600)
        histoName = "hLandauP_det{}".format(i)

        histo = ROOT.TH1D(histoName,
                          "Deposicao de energia de protoes - Detetor {} (fit Landau);Energia (MeV);Contagens".format(i),
                          nBins, minBin, maxBin)

        dados.Draw("{}>>{}".format(exprMeV, histoName), selecao, "goff")

        histo.SetLineColor(ROOT.kMagenta + 1)
        histo.SetFillColor(ROOT.kMagenta - 9)
        histo.SetLineWidth(1)

        landauFit = ROOT.TF1("landauFitP_{}".format(i), "landau", 1.0, 15.0)
        landauFit.SetLineColor(ROOT.kBlack)
        landauFit.SetLineWidth(2)
        histo.Fit(landauFit, "RQ")

        mpv   = landauFit.GetParameter(1)
        sigma = landauFit.GetParameter(2)

        histo.Draw("HIST")
        landauFit.Draw("SAME")

        canvas.SaveAs("{}/landau_protoes_detector{}.png".format(OUTPUT_DIR, i))
        canvas.Close()

        print("  Detetor {}: MPV = {:.3f} +/- {:.3f} MeV | Sigma = {:.3f} +/- {:.3f} MeV".format(
            i, mpv, landauFit.GetParError(1), sigma, landauFit.GetParError(2)))

    ROOT.gStyle.SetOptStat(1)
    ROOT.gStyle.SetOptFit(0)
    print("[OK] plot_landau_fit_protons")


def plot_landau_fit_kaons():
    """
    Ajuste de uma distribuicao de Landau a deposicao de energia de kaoes em cada detetor.
    Filtros: momentum_GeV > 0.5 e EdepDet > 10 keV.
    Kaoes (massa 494 MeV/c^2) sao intermendios entre pioes (140 MeV/c^2) e
    protoes (938 MeV/c^2) — o MPV deve ser intermedio, confirmando Bethe-Bloch.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    KEV_TO_MEV = 1e-3

    dados = carregar_chain("tracksData")

    nBins  = 300
    minBin = 0.0
    maxBin = 12.0  # MeV — kaoes depositam mais que pioes, menos que protoes

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(1)

    for i in range(4):
        branchName = "EdepDet{}_keV".format(i)
        exprMeV    = "{}*{}".format(branchName, KEV_TO_MEV)
        selecao    = "(particlePDG==321 || particlePDG==-321) && {}>10 && momentum_GeV>0.5".format(branchName)

        canvas    = ROOT.TCanvas("c_landau_k_{}".format(i), "", 800, 600)
        histoName = "hLandauK_det{}".format(i)

        histo = ROOT.TH1D(histoName,
                          "Deposicao de energia de kaoes - Detetor {} (fit Landau);Energia (MeV);Contagens".format(i),
                          nBins, minBin, maxBin)

        dados.Draw("{}>>{}".format(exprMeV, histoName), selecao, "goff")

        histo.SetLineColor(ROOT.kCyan + 1)
        histo.SetFillColor(ROOT.kCyan - 9)
        histo.SetLineWidth(1)

        landauFit = ROOT.TF1("landauFitK_{}".format(i), "landau", 0.5, 8.0)
        landauFit.SetLineColor(ROOT.kBlack)
        landauFit.SetLineWidth(2)
        histo.Fit(landauFit, "RQ")

        mpv   = landauFit.GetParameter(1)
        sigma = landauFit.GetParameter(2)

        histo.Draw("HIST")
        landauFit.Draw("SAME")

        canvas.SaveAs("{}/landau_kaoes_detector{}.png".format(OUTPUT_DIR, i))
        canvas.Close()

        print("  Detetor {}: MPV = {:.3f} +/- {:.3f} MeV | Sigma = {:.3f} +/- {:.3f} MeV".format(
            i, mpv, landauFit.GetParError(1), sigma, landauFit.GetParError(2)))

    ROOT.gStyle.SetOptStat(1)
    ROOT.gStyle.SetOptFit(0)
    print("[OK] plot_landau_fit_kaons")


def plot_mpv_vs_mass():
    """
    MPV do fit Landau em funcao da massa da particula (piao, kaon, protao).
    Confirmacao directa da formula de Bethe-Bloch: partículas mais pesadas
    depositam mais energia (MPV mais alto). Usa os valores do detetor 0.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    KEV_TO_MEV = 1e-3

    dados = carregar_chain("tracksData")

    # (nome, massa em MeV/c^2, selecao PDG, range fit, cor)
    especies = [
        ("Piao",   139.6,  "(particlePDG==211 || particlePDG==-211)", (0.3, 5.0),  ROOT.kRed+1),
        ("Kaon",   493.7,  "(particlePDG==321 || particlePDG==-321)", (0.5, 8.0),  ROOT.kCyan+1),
        ("Protao", 938.3,  "particlePDG==2212",                       (1.0, 15.0), ROOT.kMagenta+1),
    ]

    massas   = []
    mpv_vals = []
    mpv_errs = []
    nomes    = []
    cores    = []

    nBins  = 300
    minBin = 0.0

    for nome, massa, selecaoPDG, fitRange, cor in especies:
        branchName = "EdepDet0_keV"
        exprMeV    = "{}*{}".format(branchName, KEV_TO_MEV)
        selecao    = "{} && {}>10 && momentum_GeV>0.5".format(selecaoPDG, branchName)
        maxBin     = fitRange[1] * 2.5

        histoName = "hMass_{}".format(nome)
        histo = ROOT.TH1D(histoName, "", nBins, minBin, maxBin)
        dados.Draw("{}>>{}".format(exprMeV, histoName), selecao, "goff")

        landauFit = ROOT.TF1("landauMass_{}".format(nome), "landau", fitRange[0], fitRange[1])
        histo.Fit(landauFit, "RQ0")

        massas.append(massa)
        mpv_vals.append(landauFit.GetParameter(1))
        mpv_errs.append(landauFit.GetParError(1))
        nomes.append(nome)
        cores.append(cor)

    canvas = ROOT.TCanvas("c_mpv_mass", "", 800, 600)
    canvas.SetGrid()

    graf = ROOT.TGraphErrors(len(massas))
    for i in range(len(massas)):
        graf.SetPoint(i, massas[i], mpv_vals[i])
        graf.SetPointError(i, 0.0, mpv_errs[i])

    graf.SetTitle("MPV do fit Landau vs massa da particula (detetor 0);Massa (MeV/c^{2});MPV (MeV)")
    graf.SetMarkerStyle(21)
    graf.SetMarkerSize(1.8)
    graf.SetMarkerColor(ROOT.kBlack)
    graf.SetLineColor(ROOT.kBlack)
    graf.SetLineWidth(2)

    graf.Draw("AP")
    graf.GetXaxis().SetLimits(0, 1100)
    graf.SetMinimum(1.3)
    graf.SetMaximum(2.1)

    # ajuste de lei de potencia MPV ~ a * m^b como aproximacao da dependencia
    # em massa de Bethe-Bloch a momento fixo (p > 0.5 GeV/c):
    # particulas mais pesadas têm beta menor → 1/beta^2 maior → mais ionizacao
    bbApprox = ROOT.TF1("bbApprox", "[0]*pow(x,[1])", 100.0, 1050.0)
    bbApprox.SetParameter(0, 1.0)
    bbApprox.SetParameter(1, 0.2)
    bbApprox.SetLineColor(ROOT.kBlue + 1)
    bbApprox.SetLineWidth(2)
    bbApprox.SetLineStyle(2)  # tracejado para distinguir dos pontos experimentais
    graf.Fit(bbApprox, "RQ")
    bbApprox.Draw("SAME")

    # adicionar etiquetas com o nome de cada ponto
    labels = []
    offsets = [(30, 0.04), (30, 0.04), (30, 0.04)]
    for i, (nome, massa, mpv) in enumerate(zip(nomes, massas, mpv_vals)):
        dx, dy = offsets[i]
        lbl = ROOT.TLatex(massa + dx, mpv + dy, nome)
        lbl.SetTextSize(0.04)
        lbl.Draw()
        labels.append(lbl)

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)
    canvas.SaveAs("{}/mpv_vs_massa.png".format(OUTPUT_DIR))
    canvas.Close()

    ROOT.gStyle.SetOptStat(1)
    print("[OK] plot_mpv_vs_mass")
    for nome, massa, mpv, err in zip(nomes, massas, mpv_vals, mpv_errs):
        print("  {:8s} ({:.1f} MeV/c^2): MPV = {:.4f} +/- {:.4f} MeV".format(
            nome, massa, mpv, err))


def plot_mpv_pioes_vs_protoes():
    """
    Comparacao do MPV do fit Landau entre pioes e protoes em funcao do detetor.
    Mostra a diferenca de perda de energia entre as duas especies num unico grafico,
    evidenciando a dependencia em massa prevista pela formula de Bethe-Bloch.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    KEV_TO_MEV = 1e-3

    dados = carregar_chain("tracksData")

    nBins  = 300
    minBin = 0.0

    particulas = [
        ("Pioes",   ROOT.kRed+1,     ROOT.kRed+1,     "(particlePDG==211 || particlePDG==-211)", 8.0,  21),
        ("Protoes", ROOT.kMagenta+1, ROOT.kMagenta+1, "particlePDG==2212",                       20.0, 22),
    ]

    canvas = ROOT.TCanvas("c_mpv_compare", "", 800, 600)
    canvas.SetGrid()

    legenda = ROOT.TLegend(0.65, 0.72, 0.88, 0.88)
    legenda.SetBorderSize(1)

    # primeiro recolher todos os MPV (sem desenhar nada)
    graficos = []
    for nome, corM, corL, selecaoPDG, maxBin, marcador in particulas:
        mpv_vals = []
        mpv_errs = []

        for i in range(4):
            branchName = "EdepDet{}_keV".format(i)
            exprMeV    = "{}*{}".format(branchName, KEV_TO_MEV)
            selecao    = "{} && {}>10 && momentum_GeV>0.5".format(selecaoPDG, branchName)

            histoName = "hCmp_{}_{}".format(nome, i)
            histo = ROOT.TH1D(histoName, "", nBins, minBin, maxBin)
            dados.Draw("{}>>{}".format(exprMeV, histoName), selecao, "goff")  # goff = sem desenhar

            landauFit = ROOT.TF1("landauCmp_{}_{}".format(nome, i), "landau", 0.3, maxBin * 0.7)
            histo.Fit(landauFit, "RQ0")  # 0 = nao desenhar o fit

            mpv_vals.append(landauFit.GetParameter(1))
            mpv_errs.append(landauFit.GetParError(1))

        graf = ROOT.TGraphErrors(4)
        for i in range(4):
            graf.SetPoint(i, i, mpv_vals[i])
            graf.SetPointError(i, 0.0, mpv_errs[i])

        graf.SetMarkerStyle(marcador)
        graf.SetMarkerSize(1.5)
        graf.SetMarkerColor(corM)
        graf.SetLineColor(corL)
        graf.SetLineWidth(2)

        ajuste = ROOT.TF1("ajuste_{}".format(nome), "pol1", -0.5, 3.5)
        ajuste.SetLineColor(corL)
        ajuste.SetLineWidth(1)
        ajuste.SetLineStyle(2)
        graf.Fit(ajuste, "RQ")

        legenda.AddEntry(graf, nome, "p")
        graficos.append((graf, ajuste))

    # agora desenhar tudo no canvas limpo
    canvas.cd()
    for k, (graf, ajuste) in enumerate(graficos):
        opcao = "AP" if k == 0 else "P SAME"
        graf.Draw(opcao)
        ajuste.Draw("SAME")

    graficos[0][0].SetTitle("MPV do fit Landau — pioes vs protoes;Numero do detetor;MPV (MeV)")
    graficos[0][0].GetXaxis().SetLimits(-0.5, 3.5)
    graficos[0][0].GetXaxis().SetNdivisions(4)
    graficos[0][0].GetYaxis().SetTitleOffset(1.3)
    # ajustar eixo Y para incluir ambas as especies
    graficos[0][0].SetMinimum(1.2)
    graficos[0][0].SetMaximum(2.1)

    ROOT.gStyle.SetOptFit(0)
    ROOT.gStyle.SetOptStat(0)
    legenda.Draw()
    canvas.SaveAs("{}/mpv_pioes_vs_protoes.png".format(OUTPUT_DIR))
    canvas.Close()

    ROOT.gStyle.SetOptStat(1)
    print("[OK] plot_mpv_pioes_vs_protoes")


def plot_detectors_overlay():
    """
    Sobreposicao dos histogramas de deposicao de energia dos 4 detetores num so grafico.
    Permite comparar directamente o comportamento de cada detetor e verificar
    se existem diferencas sistematicas entre eles.
    Tree: edep_Per_Event, campos detector0..3 (em eV) — convertidos para MeV.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dados = carregar_chain("edep_Per_Event")

    nBins = 200
    minBin = 0.0
    maxBin = 0.6  # MeV

    cores = [ROOT.kBlue+1, ROOT.kRed+1, ROOT.kGreen+2, ROOT.kOrange+1]

    canvas = ROOT.TCanvas("c_overlay", "", 800, 600)
    canvas.SetLogy()

    legenda = ROOT.TLegend(0.65, 0.65, 0.88, 0.88)
    legenda.SetBorderSize(1)

    histos = []
    for i in range(4):
        branchName = "detector{}".format(i)
        histoName  = "hOverlay_det{}".format(i)

        histo = ROOT.TH1D(histoName,
                          "Deposicao de energia — comparacao entre detetores;Energia (MeV);Contagens",
                          nBins, minBin, maxBin)

        dados.Draw("{}*{}>>{}" .format(branchName, EV_TO_MEV, histoName),
                   "{}>0".format(branchName), "goff")

        histo.SetLineColor(cores[i])
        histo.SetLineWidth(2)

        opcao = "HIST" if i == 0 else "HIST SAME"
        histo.Draw(opcao)
        legenda.AddEntry(histo, "Detetor {}".format(i), "l")
        histos.append(histo)

    # ajustar eixo Y ao maximo
    maximo = max(h.GetMaximum() for h in histos)
    histos[0].SetMaximum(maximo * 3)
    histos[0].SetMinimum(0.5)

    ROOT.gStyle.SetOptStat(0)
    legenda.Draw()
    canvas.SaveAs("{}/deposicao_energia_4detetores.png".format(OUTPUT_DIR))
    canvas.Close()

    ROOT.gStyle.SetOptStat(1)
    print("[OK] plot_detectors_overlay")


def plot_dedx_vs_momentum():
    """
    Grafico de dE/dx (deposicao de energia no detetor 0) vs momento para
    pioes, kaoes e protoes — bandas de Bethe-Bloch para identificacao de
    particulas (PID) por perda de energia.

    Cada especie ocupa uma banda distinta a baixo momento; as bandas convergem
    a alto momento (regime ultrarelativista), onde a separacao por dE/dx deixa
    de ser praticavel.

    Usa bins logaritmicos em momento e TProfile (valor medio por bin) para
    mostrar a curva de Bethe-Bloch de cada especie.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    KEV_TO_MEV = 1e-3
    dados = carregar_chain("tracksData")

    especies = [
        ("Pioes",   "#pi^{#pm}", ROOT.kRed+1,     "(particlePDG==211||particlePDG==-211)"),
        ("Kaoes",   "K^{#pm}",   ROOT.kCyan+2,    "(particlePDG==321||particlePDG==-321)"),
        ("Protoes", "p",          ROOT.kMagenta+1, "particlePDG==2212"),
    ]

    # bins logaritmicos: 0.2 a 15 GeV/c em 50 intervalos
    nBinsMom = 50
    momMin, momMax = 0.2, 15.0
    logBins = arr.array('d', [
        10**(math.log10(momMin) + i * (math.log10(momMax) - math.log10(momMin)) / nBinsMom)
        for i in range(nBinsMom + 1)
    ])
    nBinsE, eMin, eMax = 200, 0.05, 20.0

    canvas = ROOT.TCanvas("c_dedx_mom", "", 900, 650)
    canvas.SetLogx()
    canvas.SetLogy()
    canvas.SetGrid()
    canvas.SetLeftMargin(0.12)

    legenda = ROOT.TLegend(0.15, 0.65, 0.42, 0.88)
    legenda.SetBorderSize(1)

    profiles = []
    h2list = []  # manter referencias para evitar garbage collection

    for k, (nome, latex_nome, cor, selecao) in enumerate(especies):
        sel = "{} && EdepDet0_keV>10 && momentum_GeV>{} && momentum_GeV<{}".format(
            selecao, momMin, momMax)

        histoName = "h2dedx_{}".format(nome)
        h2 = ROOT.TH2D(histoName, "", nBinsMom, logBins, nBinsE, eMin, eMax)
        dados.Draw("EdepDet0_keV*{}:momentum_GeV>>{}".format(KEV_TO_MEV, histoName),
                   sel, "goff")

        prof = h2.ProfileX("prof_dedx_{}".format(nome))
        prof.SetLineColor(cor)
        prof.SetMarkerColor(cor)
        prof.SetMarkerStyle(20 + k)
        prof.SetMarkerSize(0.9)
        prof.SetLineWidth(2)

        opcao = "E" if k == 0 else "E SAME"
        prof.Draw(opcao)
        legenda.AddEntry(prof, latex_nome, "lp")
        profiles.append(prof)
        h2list.append(h2)

    profiles[0].SetTitle(
        "Perda de energia vs momento (detetor 0);"
        "p (GeV/c);#langle dE/dx #rangle (MeV)")
    profiles[0].GetYaxis().SetRangeUser(0.3, 15.0)
    profiles[0].GetXaxis().SetMoreLogLabels()

    ROOT.gStyle.SetOptStat(0)
    legenda.Draw()
    canvas.SaveAs("{}/dedx_vs_momentum.png".format(OUTPUT_DIR))
    canvas.Close()
    ROOT.gStyle.SetOptStat(1)
    print("[OK] plot_dedx_vs_momentum")


def plot_resolution_per_species():
    """
    Resolucao relativa do detetor: sigma/MPV do fit de Landau em funcao do
    numero do detetor, para pioes, kaoes e protoes.

    sigma/MPV mede a largura relativa da distribuicao de Landau — valores mais
    baixos indicam melhor capacidade de identificacao de particulas (PID).

    Tambem calcula o poder de separacao entre pioes e protoes:
        N_sigma = |MPV_p - MPV_pi| / sqrt(sigma_p^2 + sigma_pi^2)
    O criterio padrao de PID eficiente e N_sigma > 3.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    KEV_TO_MEV = 1e-3
    dados = carregar_chain("tracksData")

    especies = [
        ("Pioes",   ROOT.kRed+1,     "(particlePDG==211||particlePDG==-211)", 8.0,  0.3, 5.0,  21),
        ("Kaoes",   ROOT.kCyan+2,    "(particlePDG==321||particlePDG==-321)", 12.0, 0.5, 8.0,  22),
        ("Protoes", ROOT.kMagenta+1, "particlePDG==2212",                     20.0, 1.0, 15.0, 23),
    ]

    nBins = 300
    canvas = ROOT.TCanvas("c_resolucao", "", 800, 600)
    canvas.SetGrid()

    legenda = ROOT.TLegend(0.65, 0.65, 0.88, 0.88)
    legenda.SetBorderSize(1)

    resultados = {}
    graficos   = []

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)

    for k, (nome, cor, selecao, maxBin, fitLow, fitHigh, marcador) in enumerate(especies):
        resolucoes    = []
        resolucao_err = []
        mpv_vals      = []
        sigma_vals    = []

        for i in range(4):
            branchName = "EdepDet{}_keV".format(i)
            exprMeV    = "{}*{}".format(branchName, KEV_TO_MEV)
            sel = "{} && {}>10 && momentum_GeV>0.5".format(selecao, branchName)

            histoName = "hRes_{}_{}".format(nome, i)
            histo = ROOT.TH1D(histoName, "", nBins, 0.0, maxBin)
            dados.Draw("{}>>{}".format(exprMeV, histoName), sel, "goff")

            if histo.GetEntries() < 50:
                resolucoes.append(0)
                resolucao_err.append(0)
                mpv_vals.append(0)
                sigma_vals.append(0)
                continue

            landauFit = ROOT.TF1("landauRes_{}_{}".format(nome, i), "landau", fitLow, fitHigh)
            histo.Fit(landauFit, "RQ0")

            mpv       = landauFit.GetParameter(1)
            sigma     = landauFit.GetParameter(2)
            mpv_err_v = landauFit.GetParError(1)
            sig_err_v = landauFit.GetParError(2)

            if mpv > 0 and sigma > 0:
                res     = sigma / mpv
                res_err = res * ((sig_err_v / sigma)**2 + (mpv_err_v / mpv)**2)**0.5
            else:
                res, res_err = 0.0, 0.0

            resolucoes.append(res)
            resolucao_err.append(res_err)
            mpv_vals.append(mpv)
            sigma_vals.append(sigma)

        resultados[nome] = {"mpv": mpv_vals, "sigma": sigma_vals}

        graf = ROOT.TGraphErrors(4)
        for i in range(4):
            graf.SetPoint(i, i, resolucoes[i])
            graf.SetPointError(i, 0.0, resolucao_err[i])

        graf.SetMarkerStyle(marcador)
        graf.SetMarkerSize(1.5)
        graf.SetMarkerColor(cor)
        graf.SetLineColor(cor)
        graf.SetLineWidth(2)

        opcao = "AP" if k == 0 else "P SAME"
        graf.Draw(opcao)
        legenda.AddEntry(graf, nome, "p")
        graficos.append(graf)

    graficos[0].SetTitle(
        "#sigma/MPV por especie vs numero do detetor;"
        "Numero do detetor;#sigma / MPV")
    graficos[0].GetXaxis().SetLimits(-0.5, 3.5)
    graficos[0].GetXaxis().SetNdivisions(4)
    graficos[0].SetMinimum(0.0)
    graficos[0].SetMaximum(0.30)

    legenda.Draw()
    canvas.SaveAs("{}/resolucao_por_especie.png".format(OUTPUT_DIR))
    canvas.Close()

    print("[OK] plot_resolution_per_species")
    print("  Resolucao sigma/MPV por especie:")
    for nome in resultados:
        mpvs   = resultados[nome]["mpv"]
        sigmas = resultados[nome]["sigma"]
        for i in range(4):
            if mpvs[i] > 0:
                print("    {:8s} det {}: sigma/MPV = {:.4f}".format(
                    nome, i, sigmas[i] / mpvs[i]))

    print("  Poder de separacao pi vs p (N_sigma = |MPV_p - MPV_pi| / sqrt(s_p^2 + s_pi^2)):")
    for i in range(4):
        mp_pi = resultados.get("Pioes",   {}).get("mpv",   [0]*4)[i]
        mp_p  = resultados.get("Protoes", {}).get("mpv",   [0]*4)[i]
        s_pi  = resultados.get("Pioes",   {}).get("sigma", [0]*4)[i]
        s_p   = resultados.get("Protoes", {}).get("sigma", [0]*4)[i]
        if mp_pi > 0 and mp_p > 0 and s_pi > 0 and s_p > 0:
            sep = abs(mp_p - mp_pi) / (s_pi**2 + s_p**2)**0.5
            print("    Detetor {}: {:.2f} sigma".format(i, sep))

    ROOT.gStyle.SetOptStat(1)


if __name__ == "__main__":
    plot_beam_composition()
    plot_energy_deposition_per_detector()
    plot_energy_per_particle_per_detector()
    plot_total_energy_loss_per_particle()
    plot_landau_fit_per_detector()
    plot_mpv_vs_detector()
    plot_landau_fit_protons()
    plot_landau_fit_kaons()
    plot_mpv_vs_mass()
    plot_mpv_pioes_vs_protoes()
    plot_detectors_overlay()
    plot_dedx_vs_momentum()
    plot_resolution_per_species()
