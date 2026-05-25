void fitLandau(){

TFile *ficheiro=new TFile("AmberTarget_Run_0.root","READ");
TFile *ficheiroGravar=new TFile("Analise.root","RECREATE");
TTree *dados=(TTree*)ficheiro->Get("tracksData");

Int_t nBins=1000;
Double_t minBin=0.0;
Double_t maxBin=30000;

TH1D* histoDetetor=new TH1D("histoDetetor","histoDetetor",nBins,minBin,maxBin);

dados->Draw("EdepDet0_keV>>histoDetetor","(particlePDG== 211 || particlePDG==-211) && EdepDet0_keV>10 && momentum_GeV>0.5","goff");
// dados->Draw("EdepDet0_keV>>histoDetetor","(particlePDG== 211 || particlePDG==-211) && EdepDet0_keV>10 ","goff");

histoDetetor->Draw();

TF1 *landauFit = new TF1("landauFit","landau",0,30000);


histoDetetor->Fit("landauFit");

histoDetetor->SetTitle("TECII");
gPad->Update();
histoDetetor->Write();

}   
