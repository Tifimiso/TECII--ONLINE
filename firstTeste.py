import ROOT

myFile=ROOT.TFile("AmberTarget_Run_0.root","READ")
#browser=ROOT.TBrowser()
myFile.ls()
#beamData=myFile.Get("BeamData")
#beamData.Print()

vertexTree=myFile.Get("hadronicVertex")
#vertexTree.Print()
#vertexTree.Draw("vertexPosZ_cm")
histogram=ROOT.TH1D("Vertex","Vertex",400,-400,0) 
histogram2=ROOT.TH1D("primaryVertex","primaryVertex",400,-400,0) 
#vertexTree.Draw("vertexPosZ_cm>>primaryVertex","","SAME")   
vertexTree.Draw("vertexPosZ_cm>>Vertex","","goff")
vertexTree.Draw("vertexPosZ_cm>>primaryVertex","IsPrimary==1","goff") 
#vertexTree.Draw("vertexPosZ_cm>>primaryVertex","IsPrimary==1","goff")
histogram.SetLineColor(ROOT.kRed)
histogram2.SetLineColor(ROOT.kBlue)
histogram.Draw()
histogram2.Draw("SAME")

ROOT.gStyle.SetOpStat(0)

# h = ROOT.TH1F("myHist", "myTitle", 64, -4, 4)
# h.FillRandom("gaus")
# h.Draw()
#dar control f (supostamente) para o histograma