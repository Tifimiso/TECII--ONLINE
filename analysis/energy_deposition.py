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
import sys
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


def verificar_ficheiros():
    """
    Verifica que todos os ficheiros ROOT estao presentes antes de iniciar a analise.
    Termina com mensagem clara se algum ficheiro estiver em falta.
    """
    em_falta = [f for f in DATA_FILES if not os.path.isfile(f)]
    if em_falta:
        print("[ERRO] Ficheiros ROOT em falta:")
        for f in em_falta:
            print("  - {}".format(f))
        print()
        print("Instrucoes: copiar os ficheiros .root para a pasta data/")
        print("Ver data/README.md para mais informacoes.")
        sys.exit(1)
    print("[OK] Todos os {} ficheiros ROOT encontrados.".format(len(DATA_FILES)))


def carregar_chain(tree_name):
    """
    Carrega uma TTree de todos os runs numa TChain.
    Verifica que a chain nao esta vazia antes de devolver.
    """
    chain = ROOT.TChain(tree_name)
    for f in DATA_FILES:
        chain.Add(f)
    n = chain.GetEntries()
    if n == 0:
        print("[AVISO] TChain '{}' esta vazia — nenhum evento encontrado.".format(tree_name))
        print("        Verifique se o nome da TTree e correcto e se os ficheiros ROOT sao validos.")
    else:
        print("[OK] TChain '{}' carregada: {:,} entradas.".format(tree_name, n))
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
    canvas.SetBottomMargin(0.18)

    n = len(nomes)
    hbar = ROOT.TH1D("hBeam",
                     "Composicao do feixe (4 runs);Especie de particula;Numero de tracks",
                     n, 0, n)

    for i, (nome, count, cor) in enumerate(zip(nomes, counts, cores)):
        hbar.GetXaxis().SetBinLabel(i + 1, nome)
        hbar.SetBinContent(i + 1, count)

    hbar.SetFillColor(ROOT.kBlue - 9)
    hbar.SetLineColor(ROOT.kBlue + 1)
    hbar.GetXaxis().SetLabelSize(0.05)
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

    graf.Draw("APL")
    graf.GetXaxis().SetLimits(0, 1100)
    graf.SetMinimum(1.3)
    graf.SetMaximum(2.1)

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

    # fitLow e fitHigh consistentes com plot_landau_fit_per_detector e plot_resolution_per_species
    particulas = [
        ("Pioes",   ROOT.kRed+1,     ROOT.kRed+1,     "(particlePDG==211 || particlePDG==-211)", 8.0,  0.3, 5.0,  21),
        ("Protoes", ROOT.kMagenta+1, ROOT.kMagenta+1, "particlePDG==2212",                       20.0, 1.0, 15.0, 22),
    ]

    canvas = ROOT.TCanvas("c_mpv_compare", "", 800, 600)
    canvas.SetGrid()

    legenda = ROOT.TLegend(0.65, 0.72, 0.88, 0.88)
    legenda.SetBorderSize(1)

    # primeiro recolher todos os MPV (sem desenhar nada)
    graficos = []
    for nome, corM, corL, selecaoPDG, maxBin, fitLow, fitHigh, marcador in particulas:
        mpv_vals = []
        mpv_errs = []

        for i in range(4):
            branchName = "EdepDet{}_keV".format(i)
            exprMeV    = "{}*{}".format(branchName, KEV_TO_MEV)
            selecao    = "{} && {}>10 && momentum_GeV>0.5".format(selecaoPDG, branchName)

            histoName = "hCmp_{}_{}".format(nome, i)
            histo = ROOT.TH1D(histoName, "", nBins, minBin, maxBin)
            dados.Draw("{}>>{}".format(exprMeV, histoName), selecao, "goff")  # goff = sem desenhar

            landauFit = ROOT.TF1("landauCmp_{}_{}".format(nome, i), "landau", fitLow, fitHigh)
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
        "p (GeV/c);")
    profiles[0].GetYaxis().SetTitle("#LT dE/dx #GT (MeV)")
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

    pares = [
        ("pi vs K", "Pioes",   "Kaoes"),
        ("K  vs p", "Kaoes",   "Protoes"),
        ("pi vs p", "Pioes",   "Protoes"),
    ]
    for label, esp1, esp2 in pares:
        print("  Poder de separacao {} (N_sigma = |MPV2-MPV1| / sqrt(s1^2+s2^2)):".format(label))
        for i in range(4):
            mp1 = resultados.get(esp1, {}).get("mpv",   [0]*4)[i]
            mp2 = resultados.get(esp2, {}).get("mpv",   [0]*4)[i]
            s1  = resultados.get(esp1, {}).get("sigma", [0]*4)[i]
            s2  = resultados.get(esp2, {}).get("sigma", [0]*4)[i]
            if mp1 > 0 and mp2 > 0 and s1 > 0 and s2 > 0:
                sep = abs(mp2 - mp1) / (s1**2 + s2**2)**0.5
                print("    Detetor {}: {:.3f} sigma".format(i, sep))

    ROOT.gStyle.SetOptStat(1)


# ---------------------------------------------------------------------------
# helper: formula de Bethe-Bloch para silicio
# ---------------------------------------------------------------------------

def _bethe_bloch_dedx(p_gev, mass_gev):
    """
    Perda de energia media (Bethe-Bloch) em silicio para particula de carga z=1.
    Devolve dE/dx em MeV*g^-1*cm^2 (sem correccao de densidade).
    Parametros do silicio (PDG 2022): Z=14, A=28.09, I=173 eV.
    """
    K     = 0.307075   # MeV mol^-1 cm^2
    me_c2 = 0.511      # MeV
    Z, A  = 14, 28.09
    I     = 173.0e-6   # MeV
    M     = mass_gev * 1000.0
    p     = p_gev  * 1000.0
    if p <= 0 or M <= 0:
        return 0.0
    bg    = p / M
    gamma = math.sqrt(1.0 + bg * bg)
    beta2 = bg * bg / (1.0 + bg * bg)
    Tmax  = (2.0 * me_c2 * bg * bg) / (
        1.0 + 2.0 * gamma * me_c2 / M + (me_c2 / M) ** 2)
    if beta2 <= 0 or Tmax <= 0:
        return 0.0
    arg = 2.0 * me_c2 * bg * bg * Tmax / (I * I)
    if arg <= 1.0:
        return 0.0
    dedx = K * (Z / A) / beta2 * (0.5 * math.log(arg) - beta2)
    return max(dedx, 0.0)


def plot_landau_overlay():
    """
    Sobreposicao das distribuicoes de Landau ajustadas para pioes, kaoes e protoes
    no detetor 0, normalizadas ao pico de cada especie.
    Permite comparar directamente a posicao do MPV e a largura de cada distribuicao
    num unico grafico — visualizacao classica de PID por dE/dx em fisica de particulas.
    Evidencia que pioes e kaoes têm MPV muito proximos (separacao dificil) enquanto
    os protoes estao claramente deslocados para energias superiores.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    KEV_TO_MEV = 1e-3
    dados = carregar_chain("tracksData")

    especies = [
        ("Pioes",   "#pi^{#pm}", ROOT.kRed+1,     "(particlePDG==211||particlePDG==-211)", 8.0,  0.3, 5.0),
        ("Kaoes",   "K^{#pm}",   ROOT.kCyan+2,    "(particlePDG==321||particlePDG==-321)", 12.0, 0.5, 8.0),
        ("Protoes", "p",          ROOT.kMagenta+1, "particlePDG==2212",                     20.0, 1.0, 15.0),
    ]

    nBins      = 300
    branchName = "EdepDet0_keV"
    exprMeV    = "{}*{}".format(branchName, KEV_TO_MEV)

    canvas = ROOT.TCanvas("c_landau_overlay", "", 900, 600)
    canvas.SetLogy()
    canvas.SetGrid()
    canvas.SetLeftMargin(0.12)

    legenda = ROOT.TLegend(0.55, 0.65, 0.88, 0.88)
    legenda.SetBorderSize(1)

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)

    histos    = []
    fits_list = []

    for k, (nome, latex_nome, cor, selecaoPDG, maxBin, fitLow, fitHigh) in enumerate(especies):
        selecao   = "{} && {}>10 && momentum_GeV>0.5".format(selecaoPDG, branchName)
        histoName = "hOvl_{}".format(nome)
        histo = ROOT.TH1D(histoName, "", nBins, 0.0, maxBin)
        dados.Draw("{}>>{}".format(exprMeV, histoName), selecao, "goff")

        landauFit = ROOT.TF1("landauOvl_{}".format(nome), "landau", fitLow, fitHigh)
        landauFit.SetLineColor(cor)
        landauFit.SetLineWidth(2)
        landauFit.SetLineStyle(2)
        histo.Fit(landauFit, "RQ0")
        mpv = landauFit.GetParameter(1)

        # normalizar pelo pico para comparar formas (MPV nao depende da normalizacao)
        peak = histo.GetMaximum()
        if peak > 0:
            histo.Scale(1.0 / peak)
            landauFit.SetParameter(0, landauFit.GetParameter(0) / peak)

        histo.SetLineColor(cor)
        histo.SetLineWidth(2)
        histo.SetFillStyle(0)

        opcao = "HIST" if k == 0 else "HIST SAME"
        histo.Draw(opcao)
        legenda.AddEntry(histo, "{} (MPV = {:.2f} MeV)".format(latex_nome, mpv), "l")
        histos.append(histo)
        fits_list.append(landauFit)

    for f in fits_list:
        f.Draw("SAME")

    histos[0].SetTitle(
        "Sobreposicao das distribuicoes de Landau — pioes, kaoes e protoes (detetor 0);"
        "Energia depositada (MeV);Contagens normalizadas ao pico")
    histos[0].GetXaxis().SetRangeUser(0.0, 15.0)
    histos[0].SetMaximum(3.0)
    histos[0].SetMinimum(5e-5)

    legenda.Draw()
    canvas.SaveAs("{}/landau_overlay_pi_k_p.png".format(OUTPUT_DIR))
    canvas.Close()

    ROOT.gStyle.SetOptStat(1)
    print("[OK] plot_landau_overlay")
    for k, (nome, latex_nome, cor, selecaoPDG, maxBin, fitLow, fitHigh) in enumerate(especies):
        print("  {}: MPV = {:.4f} MeV".format(nome, fits_list[k].GetParameter(1)))


def plot_run_consistency():
    """
    Verifica a consistencia dos resultados entre os 4 runs independentes.
    Ajusta a distribuicao de Landau de pioes no detetor 0 em cada run
    separadamente e compara os valores de MPV com o resultado combinado.
    Valida a estabilidade da simulacao e a reprodutibilidade dos resultados:
    runs consistentes confirmam que as flutuacoes estatisticas dominam sobre
    possiveis efeitos sistematicos entre runs.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    KEV_TO_MEV = 1e-3
    nBins      = 300
    branchName = "EdepDet0_keV"
    exprMeV    = "{}*{}".format(branchName, KEV_TO_MEV)
    selecao    = "(particlePDG==211||particlePDG==-211) && {}>10 && momentum_GeV>0.5".format(branchName)

    mpv_runs = []
    mpv_errs = []

    for i, run_file in enumerate(DATA_FILES):
        chain = ROOT.TChain("tracksData")
        chain.Add(run_file)
        histoName = "hConsist_run{}".format(i)
        histo = ROOT.TH1D(histoName, "", nBins, 0.0, 8.0)
        chain.Draw("{}>>{}".format(exprMeV, histoName), selecao, "goff")
        landauFit = ROOT.TF1("landauC_run{}".format(i), "landau", 0.3, 5.0)
        histo.Fit(landauFit, "RQ0")
        mpv_runs.append(landauFit.GetParameter(1))
        mpv_errs.append(landauFit.GetParError(1))

    # referencia: fit combinado dos 4 runs
    chain_all = carregar_chain("tracksData")
    hAll = ROOT.TH1D("hConsist_all", "", nBins, 0.0, 8.0)
    chain_all.Draw("{}>>hConsist_all".format(exprMeV), selecao, "goff")
    fitAll = ROOT.TF1("landauC_all", "landau", 0.3, 5.0)
    hAll.Fit(fitAll, "RQ0")
    mpv_ref = fitAll.GetParameter(1)
    err_ref = fitAll.GetParError(1)

    canvas = ROOT.TCanvas("c_consistency", "", 800, 600)
    canvas.SetGrid()

    n    = len(DATA_FILES)
    graf = ROOT.TGraphErrors(n)
    for i in range(n):
        graf.SetPoint(i, i, mpv_runs[i])
        graf.SetPointError(i, 0.0, mpv_errs[i])

    graf.SetTitle(
        "Consistencia do MPV de Landau entre runs (pioes, detetor 0);"
        "Numero do run;MPV (MeV)")
    graf.SetMarkerStyle(21)
    graf.SetMarkerSize(1.5)
    graf.SetMarkerColor(ROOT.kRed + 1)
    graf.SetLineColor(ROOT.kRed + 1)
    graf.SetLineWidth(2)
    graf.Draw("AP")
    graf.GetXaxis().SetLimits(-0.5, 3.5)
    graf.GetXaxis().SetNdivisions(4)
    graf.SetMinimum(mpv_ref * 0.990)
    graf.SetMaximum(mpv_ref * 1.010)

    # linha de referencia (MPV combinado) e banda de incerteza
    lineRef = ROOT.TLine(-0.5, mpv_ref, 3.5, mpv_ref)
    lineRef.SetLineColor(ROOT.kBlue + 1)
    lineRef.SetLineWidth(2)
    lineRef.SetLineStyle(2)
    lineRef.Draw()
    bandUp = ROOT.TLine(-0.5, mpv_ref + err_ref, 3.5, mpv_ref + err_ref)
    bandDn = ROOT.TLine(-0.5, mpv_ref - err_ref, 3.5, mpv_ref - err_ref)
    for line in [bandUp, bandDn]:
        line.SetLineColor(ROOT.kBlue + 1)
        line.SetLineWidth(1)
        line.SetLineStyle(3)
        line.Draw()

    legenda = ROOT.TLegend(0.38, 0.75, 0.88, 0.88)
    legenda.SetBorderSize(1)
    legenda.AddEntry(graf, "MPV por run", "p")
    legenda.AddEntry(lineRef, "MPV combinado ({:.4f} MeV)".format(mpv_ref), "l")
    legenda.Draw()

    ROOT.gStyle.SetOptStat(0)
    canvas.SaveAs("{}/consistencia_runs.png".format(OUTPUT_DIR))
    canvas.Close()
    ROOT.gStyle.SetOptStat(1)

    print("[OK] plot_run_consistency")
    print("  MPV combinado: {:.4f} +/- {:.4f} MeV".format(mpv_ref, err_ref))
    for i in range(n):
        sigma_desvio = (mpv_runs[i] - mpv_ref) / math.sqrt(mpv_errs[i]**2 + err_ref**2)
        print("  Run {}: MPV = {:.4f} +/- {:.4f} MeV  (desvio: {:+.1f} sigma)".format(
            i, mpv_runs[i], mpv_errs[i], sigma_desvio))


def plot_dedx_with_bethe_bloch():
    """
    Sobreposicao das curvas teoricas de Bethe-Bloch sobre o grafico experimental
    dE/dx vs momento, para pioes, kaoes e protoes no detetor 0.
    A espessura efectiva do detetor e inferida empiricamente a partir dos dados
    de pioes ao minimo de ionizacao (betaGamma ~ 3.5, p ~ 0.55 GeV/c).
    Parametros do silicio usados: Z=14, A=28.09, I=173 eV, rho=2.329 g/cm^3
    (PDG 2022). Sem correccao de densidade (valido para p < 10 GeV/c neste detector).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    KEV_TO_MEV = 1e-3
    RHO_SI     = 2.329  # g/cm^3
    dados = carregar_chain("tracksData")

    especies = [
        ("Pioes",   "#pi^{#pm}", ROOT.kRed+1,     "(particlePDG==211||particlePDG==-211)", 0.13957),
        ("Kaoes",   "K^{#pm}",   ROOT.kCyan+2,    "(particlePDG==321||particlePDG==-321)", 0.49368),
        ("Protoes", "p",          ROOT.kMagenta+1, "particlePDG==2212",                     0.93827),
    ]

    nBinsMom = 50
    momMin, momMax = 0.2, 15.0
    logBins = arr.array('d', [
        10 ** (math.log10(momMin) + i * (math.log10(momMax) - math.log10(momMin)) / nBinsMom)
        for i in range(nBinsMom + 1)
    ])
    nBinsE, eMin, eMax = 200, 0.05, 20.0

    canvas = ROOT.TCanvas("c_dedx_bb", "", 900, 650)
    canvas.SetLogx()
    canvas.SetLogy()
    canvas.SetGrid()
    canvas.SetLeftMargin(0.12)

    legenda = ROOT.TLegend(0.55, 0.13, 0.97, 0.53)
    legenda.SetBorderSize(1)

    profiles  = []
    h2list    = []
    graphs_bb = []  # manter referencias dos TGraph

    for k, (nome, latex_nome, cor, selecao, mass_gev) in enumerate(especies):
        sel = "{} && EdepDet0_keV>10 && momentum_GeV>{} && momentum_GeV<{}".format(
            selecao, momMin, momMax)
        histoName = "h2bb_{}".format(nome)
        h2 = ROOT.TH2D(histoName, "", nBinsMom, logBins, nBinsE, eMin, eMax)
        dados.Draw("EdepDet0_keV*{}:momentum_GeV>>{}".format(KEV_TO_MEV, histoName), sel, "goff")

        prof = h2.ProfileX("prof_bb_{}".format(nome))
        prof.SetLineColor(cor)
        prof.SetMarkerColor(cor)
        prof.SetMarkerStyle(20 + k)
        prof.SetMarkerSize(0.9)
        prof.SetLineWidth(2)

        opcao = "E" if k == 0 else "E SAME"
        prof.Draw(opcao)
        legenda.AddEntry(prof, "{} (dados)".format(latex_nome), "lp")
        profiles.append((prof, mass_gev, cor, latex_nome))
        h2list.append(h2)

    profiles[0][0].SetTitle(
        "dE/dx vs momento — comparacao com Bethe-Bloch teorico (detetor 0);"
        "p (GeV/c);")
    profiles[0][0].GetYaxis().SetTitle("#LT dE/dx #GT (MeV)")
    profiles[0][0].GetYaxis().SetRangeUser(0.3, 15.0)
    profiles[0][0].GetXaxis().SetMoreLogLabels()

    # inferir espessura efectiva do detetor a partir dos dados de pioes ao MIP
    # (betaGamma ~ 3.5 para pioes a p ~ 0.55 GeV/c)
    prof_pi  = profiles[0][0]
    p_mip    = 0.55
    bin_mip  = prof_pi.FindBin(p_mip)
    data_mip = prof_pi.GetBinContent(bin_mip)
    bb_mip   = _bethe_bloch_dedx(p_mip, 0.13957) * RHO_SI  # MeV/cm
    thickness_cm = (data_mip / bb_mip) if bb_mip > 0 else 0.04

    # curvas teoricas de Bethe-Bloch
    n_pts = 200
    for prof, mass_gev, cor, latex_nome in profiles:
        x_bb = arr.array('d')
        y_bb = arr.array('d')
        for i in range(n_pts + 1):
            p = 10 ** (math.log10(momMin) + i * (math.log10(momMax) - math.log10(momMin)) / n_pts)
            dedx = _bethe_bloch_dedx(p, mass_gev) * RHO_SI * thickness_cm
            if dedx > 0.1:
                x_bb.append(p)
                y_bb.append(dedx)
        if len(x_bb) > 1:
            g = ROOT.TGraph(len(x_bb), x_bb, y_bb)
            g.SetLineColor(cor)
            g.SetLineWidth(2)
            g.SetLineStyle(2)
            g.Draw("L SAME")
            legenda.AddEntry(g, "{} (Bethe-Bloch)".format(latex_nome), "l")
            graphs_bb.append(g)

    ROOT.gStyle.SetOptStat(0)
    legenda.Draw()
    canvas.SaveAs("{}/dedx_vs_momentum_bethe_bloch.png".format(OUTPUT_DIR))
    canvas.Close()
    ROOT.gStyle.SetOptStat(1)

    print("[OK] plot_dedx_with_bethe_bloch")
    print("  Espessura efectiva inferida dos dados: {:.3f} cm = {:.1f} mm".format(
        thickness_cm, thickness_cm * 10))


def plot_dedx_mpv_vs_momentum():
    """
    dE/dx baseado no MPV do fit de Landau por bin de momento — alternativa
    mais robusta ao TProfile (media aritmetica).

    A distribuicao de Landau e assimetrica: a cauda direita puxa a media para
    valores sistematicamente acima do MPV. O MPV e o estimador fisicamente
    correcto da perda mais provavel de energia (equacao de Bethe-Bloch).

    Para cada especie e cada bin de momento logaritmico:
      1. Extrai o histograma de deposicao de energia (detetor 0).
      2. Ajusta uma Landau na regiao do pico.
      3. Extrai o MPV e a sua incerteza do fit.
    Plota MPV vs momento como TGraphErrors via TMultiGraph, com as curvas de
    Bethe-Bloch sobrepostas para validacao directa.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    KEV_TO_MEV = 1e-3
    dados = carregar_chain("tracksData")

    # (nome, latex, cor, selecao PDG, massa GeV, limite inf fit MeV, limite sup fit MeV)
    especies = [
        ("Pioes",   "#pi^{#pm}", ROOT.kRed+1,     "(particlePDG==211||particlePDG==-211)",
         0.139570, 0.2, 3.0),
        ("Kaoes",   "K^{#pm}",   ROOT.kCyan+2,    "(particlePDG==321||particlePDG==-321)",
         0.493677, 0.3, 6.0),
        ("Protoes", "p",          ROOT.kMagenta+1, "particlePDG==2212",
         0.938272, 0.5, 10.0),
    ]

    # 18 bins logaritmicos: 0.3 a 10 GeV/c
    nBinsMom       = 18
    momMin, momMax = 0.3, 10.0
    logEdges = [
        10**(math.log10(momMin) + i * (math.log10(momMax) - math.log10(momMin)) / nBinsMom)
        for i in range(nBinsMom + 1)
    ]
    MIN_ENTRIES = 60  # minimo de entradas para fit fiavel

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)

    mg       = ROOT.TMultiGraph("mg_dedx_mpv", "")
    graphs   = []   # manter referencias Python para evitar GC
    graphs_bb = []
    legenda  = ROOT.TLegend(0.15, 0.62, 0.50, 0.88)
    legenda.SetBorderSize(1)

    # espessura efectiva inferida dos dados (calculada em plot_dedx_with_bethe_bloch)
    thickness_cm = 0.47

    for k, (nome, latex_nome, cor, selecaoPDG, mass_gev, fitLow, fitHigh) in enumerate(especies):
        x_pts  = []
        y_pts  = []
        ex_pts = []
        ey_pts = []

        for i in range(nBinsMom):
            pLow  = logEdges[i]
            pHigh = logEdges[i + 1]
            pMid  = math.sqrt(pLow * pHigh)  # centro geometrico (escala log)

            histoName = "h_mpvbin_{}_{}".format(nome, i)
            hbin = ROOT.TH1D(histoName, "", 300, 0.0, fitHigh * 3.0)

            sel = ("{} && EdepDet0_keV>5 && momentum_GeV>{:.6f} && momentum_GeV<{:.6f}"
                   .format(selecaoPDG, pLow, pHigh))
            dados.Draw("EdepDet0_keV*{}>>{}".format(KEV_TO_MEV, histoName), sel, "goff")

            nEnt = int(hbin.GetEntries())
            if nEnt < MIN_ENTRIES:
                continue

            fName   = "fLandauMPV_{}_{}".format(nome, i)
            fLandau = ROOT.TF1(fName, "landau", fitLow, fitHigh)
            # valor inicial do MPV: curva de Bethe-Bloch escalada pela espessura
            mpv0 = _bethe_bloch_dedx(pMid, mass_gev) * thickness_cm
            fLandau.SetParameter(1, mpv0)
            hbin.Fit(fLandau, "RQ0")

            mpv  = fLandau.GetParameter(1)
            empv = fLandau.GetParError(1)
            # aceitar apenas fits fisicamente plausíveis
            if empv > 0 and 0.05 < mpv < mpv0 * 5.0:
                x_pts.append(pMid)
                y_pts.append(mpv)
                ex_pts.append(0.0)
                ey_pts.append(empv)

        if len(x_pts) < 2:
            print("  [AVISO] {} — poucos pontos validos ({})".format(nome, len(x_pts)))
            continue

        ax = arr.array('d', x_pts)
        ay = arr.array('d', y_pts)
        aex = arr.array('d', ex_pts)
        aey = arr.array('d', ey_pts)
        g = ROOT.TGraphErrors(len(ax), ax, ay, aex, aey)
        g.SetName("g_dedxmpv_{}".format(nome))
        g.SetLineColor(cor)
        g.SetMarkerColor(cor)
        g.SetMarkerStyle(20 + k)
        g.SetMarkerSize(1.0)
        g.SetLineWidth(2)
        mg.Add(g, "P")
        # TMultiGraph takes ownership — dizer ao GC do Python para nao fazer delete duplo
        ROOT.SetOwnership(g, False)
        legenda.AddEntry(g, "{} (Landau MPV)".format(latex_nome), "lp")
        graphs.append(g)

        # curva de Bethe-Bloch teorica
        x_bb = arr.array('d')
        y_bb = arr.array('d')
        for ip in range(120):
            p_gev = 10**(math.log10(0.2) + ip * (math.log10(12.0) - math.log10(0.2)) / 119)
            dedx = _bethe_bloch_dedx(p_gev, mass_gev) * thickness_cm
            if 0.05 < dedx < 20.0:
                x_bb.append(p_gev)
                y_bb.append(dedx)
        if len(x_bb) > 1:
            gbb = ROOT.TGraph(len(x_bb), x_bb, y_bb)
            gbb.SetName("gbb_dedxmpv_{}".format(nome))
            gbb.SetLineColor(cor)
            gbb.SetLineWidth(2)
            gbb.SetLineStyle(2)
            graphs_bb.append(gbb)
            legenda.AddEntry(gbb, "{} (Bethe-Bloch)".format(latex_nome), "l")

    canvas = ROOT.TCanvas("c_dedx_mpv_mom", "", 900, 650)
    canvas.SetLogx()
    canvas.SetLogy()
    canvas.SetGrid()
    canvas.SetLeftMargin(0.12)

    mg.Draw("A")
    mg.SetTitle("MPV de Landau vs momento (detetor 0);p (GeV/c);MPV dE/dx (MeV)")
    mg.GetXaxis().SetMoreLogLabels()
    mg.GetYaxis().SetRangeUser(0.2, 15.0)
    canvas.Modified()
    canvas.Update()

    for gbb in graphs_bb:
        gbb.Draw("L SAME")

    legenda.Draw()
    canvas.SaveAs("{}/dedx_mpv_vs_momentum.png".format(OUTPUT_DIR))
    canvas.Close()

    ROOT.gStyle.SetOptStat(1)
    print("[OK] plot_dedx_mpv_vs_momentum")
    for k, g in enumerate(graphs):
        nome = especies[k][0]
        npts = g.GetN()
        print("  {} — {} pontos validos".format(nome, npts))


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Verificacao de prerequisitos — termina com mensagem clara se os
    # ficheiros ROOT nao estiverem presentes em data/
    # ------------------------------------------------------------------
    verificar_ficheiros()

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
    plot_run_consistency()
    plot_dedx_with_bethe_bloch()
    plot_landau_overlay()
    plot_dedx_mpv_vs_momentum()

    # ------------------------------------------------------------------
    # Confirmar outputs gerados
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  Graficos gerados em: {}".format(OUTPUT_DIR))
    print("=" * 55)
    esperados = [
        "composicao_feixe.png",
        "edep_detector0.png", "edep_detector1.png",
        "edep_detector2.png", "edep_detector3.png",
        "edep_por_particula_detector0.png",
        "edep_por_particula_detector1.png",
        "edep_por_particula_detector2.png",
        "edep_por_particula_detector3.png",
        "edep_total_por_particula.png",
        "landau_pioes_detector0.png", "landau_pioes_detector1.png",
        "landau_pioes_detector2.png", "landau_pioes_detector3.png",
        "mpv_vs_detector.png", "sigma_vs_detector.png",
        "landau_protoes_detector0.png", "landau_protoes_detector1.png",
        "landau_protoes_detector2.png", "landau_protoes_detector3.png",
        "landau_kaoes_detector0.png", "landau_kaoes_detector1.png",
        "landau_kaoes_detector2.png", "landau_kaoes_detector3.png",
        "mpv_vs_massa.png",
        "mpv_pioes_vs_protoes.png",
        "deposicao_energia_4detetores.png",
        "dedx_vs_momentum.png",
        "resolucao_por_especie.png",
        "consistencia_runs.png",
        "dedx_vs_momentum_bethe_bloch.png",
        "landau_overlay_pi_k_p.png",
        "dedx_mpv_vs_momentum.png",
    ]
    ausentes = []
    for nome in esperados:
        caminho = os.path.join(OUTPUT_DIR, nome)
        if os.path.isfile(caminho):
            print("  [OK] {}".format(nome))
        else:
            print("  [EM FALTA] {}".format(nome))
            ausentes.append(nome)
    if ausentes:
        print("\n[AVISO] {} ficheiro(s) em falta — verificar erros acima.".format(len(ausentes)))
    else:
        print("\nTodos os {} graficos gerados com sucesso.".format(len(esperados)))
    plot_resolution_per_species()
    plot_landau_overlay()
    plot_run_consistency()
    plot_dedx_with_bethe_bloch()
    plot_dedx_mpv_vs_momentum()
