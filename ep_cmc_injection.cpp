// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

/// \file ep_cmc_injection.cpp
/// \brief CMC injection into EPOS, arxiv:2605.19789
/// \author Salman Malik

#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <THnSparse.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TString.h>
#include <TTree.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

// EPOS particle info file
#include "/Users/solus/mc.codebase/eventforge/Common/epos_particle_info.h"

namespace {

constexpr Double_t _lam = 0.0015;
constexpr Double_t _eta_max = 0.5;
constexpr Double_t _pt_min = 0.2;
constexpr Double_t _pt_max = 3.0;

constexpr Double_t _mu = 1.0 / 6.0;
// Levy-walk step range in (eta,phi) units.
// _ep_rmax covers the full azimuthal window in one step.
constexpr Double_t _ep_rmax = 2. * TMath::Pi(); // ≈ 6.28
constexpr Double_t _ep_rmin = _ep_rmax * 1.0e-7;

constexpr Int_t _M_bins = 52;
constexpr Int_t _fq_ord = 4;
constexpr Int_t _fq0 = 2;

constexpr Int_t _cand_mul = 12;
constexpr Int_t _pool_min = 32;
constexpr Int_t _pool_pass = 4;
constexpr Long64_t _prog_n = 50;
constexpr Int_t _sub_size = 100; // events per subsample for error estimation

struct AccTrk {
   Int_t idx = -1;
   Double_t pt = 0.;
   Double_t px = 0.;
   Double_t py = 0.;
   Double_t pz = 0.;
};

// CMC candidate stores position in (eta,phi) space.
// Full momentum is reconstructed at injection time from (eta_cmc, phi_cmc, pT_orig).
struct CMCCand {
   Double_t eta = 0.;
   Double_t phi = 0.;
};

Double_t getPt(const Double_t px, const Double_t py) { return TMath::Sqrt(px * px + py * py); }

Double_t getEta(const Double_t pt, const Double_t pz)
{
   if (pt <= 0.) return (pz >= 0.) ? 1.e9 : -1.e9;
   const auto _r = pz / pt;
   return TMath::Log(_r + TMath::Sqrt(_r * _r + 1.));
}

Double_t getPhi(const Double_t px, const Double_t py)
{
   auto _phi = TMath::ATan2(py, px);
   if (_phi < 0.) _phi += 2. * TMath::Pi();
   return _phi;
}

Bool_t passCut(const Double_t px, const Double_t py, const Double_t pz, const Int_t pid, const Int_t ist)
{
   if (!EPOSParticleInfo::is_charged(pid))  return kFALSE;
   const auto _pt = getPt(px, py);
   if (_pt < _pt_min || _pt > _pt_max) return kFALSE;
   if (ist != 8)  return kFALSE;
   return TMath::Abs(getEta(_pt, pz)) < _eta_max;
}

// Inverse-CDF sample from rho(r) ∝ r^(-1-mu) on [rmin, rmax].
Double_t getStep(TRandom3 &_rng, const Double_t rmin, const Double_t rmax)
{
   const auto _u = TMath::Min(1. - 1.e-15, TMath::Max(1.e-15, _rng.Uniform()));
   const auto _pow = TMath::Power(rmin / rmax, _mu);
   const auto _den = TMath::Power(1. - _u * (1. - _pow), 1. / _mu);
   return rmin / _den;
}

// Levy random walk in (eta, phi) space.
// Seed: (eta0, phi0) from a randomly chosen accepted track.
// Each accepted step is stored as a CMCCand{eta, phi}.
// phi is kept in [0, 2pi); steps outside |eta| < _eta_max are rejected.
std::vector<CMCCand> makePool(const Double_t _eta0, const Double_t _phi0, const Int_t _n, TRandom3 &_rng, TH1D *_hstep)
{
   std::vector<CMCCand> _pool;
   if (_n <= 0) return _pool;

   _pool.reserve(_n);
   auto _eta = _eta0;
   auto _phi = _phi0;

   // Include the seed itself if it lies inside the window.
   if (TMath::Abs(_eta) < _eta_max) _pool.push_back({_eta, _phi});

   const auto _try_max = TMath::Max(200, _n * 200);
   for (Int_t _it = 0; _it < _try_max && static_cast<Int_t>(_pool.size()) < _n; ++_it) {
      const auto _step = getStep(_rng, _ep_rmin, _ep_rmax);
      if (_hstep) _hstep->Fill(_step);

      const auto _th = _rng.Uniform(0., 2. * TMath::Pi());
      const auto _eta1 = _eta + _step * TMath::Cos(_th);
      auto _phi1 = _phi + _step * TMath::Sin(_th);

      // Reject if outside eta acceptance.
      if (TMath::Abs(_eta1) >= _eta_max) continue;

      // Wrap phi to [0, 2pi). One correction is enough since step <= _ep_rmax = 1.0 < 2pi.
      if (_phi1 < 0.) _phi1 += 2. * TMath::Pi();
      if (_phi1 >= 2. * TMath::Pi()) _phi1 -= 2. * TMath::Pi();

      _eta = _eta1;
      _phi = _phi1;
      _pool.push_back({_eta, _phi});
   }
   return _pool;
}

void shuf(std::vector<AccTrk> &_trk, TRandom3 &_rng)
{
   if (_trk.size() < 2) return;
   for (auto _i = _trk.size() - 1; _i > 0; --_i) {
      const auto _j = static_cast<std::size_t>(_rng.Integer(_i + 1));
      std::swap(_trk[_i], _trk[_j]);
   }
}

TString getLamTag(const Double_t _x)
{
   TString _s = TString::Format("%.6f", _x);
   while (_s.EndsWith("0")) _s.Chop();
   if (_s.EndsWith(".")) _s.Chop();
   _s.ReplaceAll(".", "p");
   return _s;
}

TString stripDir(const TString &_path)
{
   const auto _pos = TMath::Max(_path.Last('/'), _path.Last('\\'));
   if (_pos < 0) return _path;
   return _path(_pos + 1, _path.Length() - _pos - 1);
}

TString stripRoot(const TString &_name)
{
   TString _out = _name;
   if (_out.EndsWith(".root")) _out.Resize(_out.Length() - 5);
   return _out;
}

TString getOutName(const TString &_in)
{
   const auto _suf = TString::Format(".cmc_lambda%s.root", getLamTag(_lam).Data());
   if (_in.EndsWith(".root")) {
      TString _out = _in;
      _out.Resize(_out.Length() - 5);
      _out += _suf;
      return _out;
   }
   return _in + _suf;
}

TString getQAName(const TString &_in)
{
   const auto _base = stripRoot(stripDir(_in));
   return TString::Format("%s.cmc_lambda%s.qa.root", _base.Data(), getLamTag(_lam).Data());
}

Int_t getNRep(const Int_t _nacc, TRandom3 &_rng)
{
   if (_nacc <= 0 || _lam <= 0.) return 0;
   const auto _p = TMath::Max(0., TMath::Min(1., _lam));
   return _rng.Binomial(_nacc, _p);
}

Int_t getMVal(const Int_t _M) { return 2 * (_M + 2); }

void setMLabel(TH1D *_h)
{
   if (!_h) return;
   for (Int_t _M = 0; _M < _M_bins; ++_M)  _h->GetXaxis()->SetBinLabel(_M + 1, Form("%d", getMVal(_M) * getMVal(_M)));
}

// Per-event raw moments: returns {fqNum[q]/M^2} and binCon/M^2 for each M bin.
// Does NOT accumulate into histograms — caller is responsible for subsample bookkeeping.
struct FqEvent {
   std::array<std::array<Double_t, _fq_ord>, _M_bins> fqPerM{};  // [M][q]
   std::array<Double_t, _M_bins> avPerM{};                        // [M]
};

FqEvent calcFqEvent(std::array<TH2D *, _M_bins> &_h2d)
{
   FqEvent _ev;
   for (Int_t _M = 0; _M < _M_bins; ++_M) {
      Double_t _sumBinCon = 0.;
      std::array<Double_t, _fq_ord> _fqNum{};
      _fqNum.fill(0.);

      for (Int_t _eb = 1; _eb <= _h2d[_M]->GetNbinsX(); ++_eb) {
         for (Int_t _pb = 1; _pb <= _h2d[_M]->GetNbinsY(); ++_pb) {
            const auto _binCon = _h2d[_M]->GetBinContent(_eb, _pb);
            _sumBinCon += _binCon;

            for (Int_t _q = 0; _q < _fq_ord; ++_q) {
               const auto _ord = _q + _fq0;
               if (_binCon < _ord) continue;
               // Falling factorial n*(n-1)*...*(n-q+1) — avoids TMath::Factorial overflow
               Double_t _fqTmp = 1.;
               for (Int_t _k = 0; _k < _ord; ++_k)  _fqTmp *= (_binCon - _k);
               _fqNum[_q] += _fqTmp;
            }
         }
      }

      const auto _squareM = TMath::Power(getMVal(_M), 2);
      _ev.avPerM[_M] = _sumBinCon / _squareM;
      for (Int_t _q = 0; _q < _fq_ord; ++_q)  _ev.fqPerM[_M][_q] = _fqNum[_q] / _squareM;
   }
   return _ev;
}

// Subsample accumulator: collects per-event raw moments, computes Fq ratio
// when a subsample completes, stores subsample ratios for final error calculation.
struct SubsampleAccum {
   // Running sums for current subsample
   std::array<std::array<Double_t, _fq_ord>, _M_bins> sumFq{};  // [M][q]
   std::array<Double_t, _M_bins> sumAv{};                        // [M]
   Int_t nEvInSub = 0;

   // Collected subsample Fq ratios and <n> for error estimation
   std::array<std::array<std::vector<Double_t>, _fq_ord>, _M_bins> subFq;  // [M][q][isub]
   std::array<std::vector<Double_t>, _M_bins> subAv;                        // [M][isub]

   void addEvent(const FqEvent &ev)
   {
      for (Int_t _M = 0; _M < _M_bins; ++_M) {
         sumAv[_M] += ev.avPerM[_M];
         for (Int_t _q = 0; _q < _fq_ord; ++_q) {
            sumFq[_M][_q] += ev.fqPerM[_M][_q];
         }
      }
      ++nEvInSub;
   }

   void closeSubsample()
   {
      if (nEvInSub <= 0) return;
      const auto norm = 1. / nEvInSub;
      for (Int_t _M = 0; _M < _M_bins; ++_M) {
         const auto avM = sumAv[_M] * norm;
         subAv[_M].push_back(avM);
         for (Int_t _q = 0; _q < _fq_ord; ++_q) {
            const auto fqM = sumFq[_M][_q] * norm;
            const auto ord = _q + _fq0;
            const auto den = TMath::Power(avM, ord);
            const auto ratio = (den > 0.) ? fqM / den : 0.;
            subFq[_M][_q].push_back(ratio);
         }
      }
      // Reset for next subsample
      for (auto &row : sumFq) row.fill(0.);
      sumAv.fill(0.);
      nEvInSub = 0;
   }

   // Write final mean ± stderr into Fq histograms, and global <n> into h_av
   void finalize(const std::array<TH1D *, _fq_ord> &hFq, TH1D *hAv = nullptr) const
   {
      for (Int_t _M = 0; _M < _M_bins; ++_M) {
         // Fill average bin content from grand mean of subsamples
         if (hAv && !subAv[_M].empty()) {
            Double_t sumA = 0.;
            for (auto v : subAv[_M]) sumA += v;
            hAv->SetBinContent(_M + 1, sumA / subAv[_M].size());
         }

         for (Int_t _q = 0; _q < _fq_ord; ++_q) {
            const auto &vals = subFq[_M][_q];
            const Int_t nSub = static_cast<Int_t>(vals.size());
            if (nSub == 0 || !hFq[_q]) continue;

            Double_t sum = 0.;
            for (auto v : vals) sum += v;
            const auto mean = sum / nSub;

            Double_t stdErr = 0.;
            if (nSub > 1) {
               Double_t sumSq = 0.;
               for (auto v : vals) sumSq += (v - mean) * (v - mean);
               stdErr = TMath::Sqrt(sumSq / (nSub - 1)) / TMath::Sqrt(nSub);
            }

            hFq[_q]->SetBinContent(_M + 1, mean);
            hFq[_q]->SetBinError(_M + 1, stdErr);
         }
      }
   }
};

} // namespace

//////////////NAME SPACE FOR FUNCTIONS//////////////////////
void ep_cmc_injection()
{
   std::ifstream _fl("files.txt");
   TString _in_name;
   if (!(_fl >> _in_name)) {
      std::cerr << "Error: Could not read filename from files.txt\n";
      return;
   }

   auto *_fin = TFile::Open(_in_name.Data(), "READ");
   if (!_fin || _fin->IsZombie()) {
      std::cerr << "Error: Could not open file " << _in_name << "\n";
      return;
   }

   auto *_tr = static_cast<TTree *>(_fin->Get("teposevent0"));
   if (!_tr) {
      std::cerr << "Error: Could not find tree 'teposevent0'\n";
      _fin->Close();
      return;
   }

   /////////////////READ BRANCHES//////////////////////
   Int_t np = 0;
   Float_t bim = 0.f;

   auto _np_max = static_cast<Int_t>(_tr->GetMaximum("np"));
   if (_np_max <= 0) _np_max = 10000;
   ++_np_max;

   std::vector<Float_t> px(_np_max), py(_np_max), pz(_np_max);
   std::vector<Int_t> id(_np_max), ist(_np_max);

   const auto _bind = [&](const char *_bn, void *_ba) {
      const auto _rc = _tr->SetBranchAddress(_bn, _ba);
      if (_rc < 0) std::cerr << "SetBranchAddress failed for '" << _bn << "' rc=" << _rc << "\n";
      return _rc;
   };

   if (_bind("np", &np) < 0 || _bind("bim", &bim) < 0 || _bind("px", px.data()) < 0 || _bind("py", py.data()) < 0 ||
       _bind("pz", pz.data()) < 0 || _bind("id", id.data()) < 0 || _bind("ist", ist.data()) < 0) {
      _fin->Close();
      return;
   }

   /////////////////CREATE OUTPUT FILES//////////////////////
   const auto _out_name = getOutName(_in_name);
   auto *_fout = TFile::Open(_out_name.Data(), "RECREATE");
   if (!_fout || _fout->IsZombie()) {
      std::cerr << "Error: Could not create output file " << _out_name << "\n";
      _fin->Close();
      return;
   }

   _fout->cd();
   auto *_tout = _tr->CloneTree(0);
   if (!_tout) {
      std::cerr << "Error: Could not clone output tree\n";
      _fout->Close();
      _fin->Close();
      return;
   }

   const auto _qa_name = getQAName(_in_name);
   auto *_fqa = TFile::Open(_qa_name.Data(), "RECREATE");
   if (!_fqa || _fqa->IsZombie()) {
      std::cerr << "Error: Could not create histogram file " << _qa_name << "\n";
      _fout->Close();
      _fin->Close();
      return;
   }

   const auto _nev = _tr->GetEntries();

   ///////////////CREATE HISTOGRAMS//////////////////////
   auto *h_nacc = new TH1D("h_nAccepted", "Accepted tracks per event;N_{accepted};Events", 400, 0., 4000.);
   auto *h_nreq = new TH1D("h_nRequested", "Requested replacements per event;N_{requested};Events", 100, 0., 100.);
   auto *h_ninj = new TH1D("h_nInjected", "Injected tracks per event;N_{injected};Events", 100, 0., 100.);
   auto *h_lam_e = new TH1D("h_lambda_event", "Event injection fraction;N_{injected}/N_{accepted};Events", 120, 0., 0.12);

   auto *h_pt_bf = new TH1D("h_pt_before", "Accepted-track p_{T} before injection;p_{T} (GeV/c);Tracks", 300, 0., _pt_max + 0.3);
   auto *h_pt_af = new TH1D("h_pt_after", "Accepted-track p_{T} after injection;p_{T} (GeV/c);Tracks", 300, 0., _pt_max + 0.3);
   auto *h_eta_bf = new TH1D("h_eta_before", "Accepted-track #eta before injection;#eta;Tracks", 120, -1.2, 1.2);
   auto *h_eta_af = new TH1D("h_eta_after", "Accepted-track #eta after injection;#eta;Tracks", 120, -1.2, 1.2);
   auto *h_phi_bf = new TH1D("h_phi_before", "Accepted-track #phi before injection;#phi;Tracks", 128, 0., 2. * TMath::Pi());
   auto *h_phi_af = new TH1D("h_phi_after", "Accepted-track #phi after injection;#phi;Tracks", 128, 0., 2. * TMath::Pi());

   auto *h_rpt_bf = new TH1D("h_replaced_pt_before", "Replaced-track p_{T} before injection;p_{T} (GeV/c);Tracks", 300, 0., _pt_max + 0.3);
   auto *h_rpt_af = new TH1D("h_replaced_pt_after", "Replaced-track p_{T} after injection;p_{T} (GeV/c);Tracks", 300, 0., _pt_max + 0.3);
   auto *h_reta_bf = new TH1D("h_replaced_eta_before", "Replaced-track #eta before injection;#eta;Tracks", 120, -1.2, 1.2);
   auto *h_reta_af = new TH1D("h_replaced_eta_after", "Replaced-track #eta after injection;#eta;Tracks", 120, -1.2, 1.2);
   auto *h_rphi_bf = new TH1D("h_replaced_phi_before", "Replaced-track #phi before injection;#phi;Tracks", 128, 0., 2. * TMath::Pi());
   auto *h_rphi_af = new TH1D("h_replaced_phi_after", "Replaced-track #phi after injection;#phi;Tracks", 128, 0., 2. * TMath::Pi());

   auto *h_dpt = new TH1D("h_deltaPt", "Matched replacement #Delta p_{T};p_{T}^{after}-p_{T}^{before} (GeV/c);Tracks", 160, -0.4, 0.4);
   auto *h_adpt = new TH1D("h_absDeltaPt", "Matched replacement |#Delta p_{T}|;|p_{T}^{after}-p_{T}^{before}| (GeV/c);Tracks", 120, 0., 0.3);
   auto *h_dpx = new TH1D("h_deltaPx", "Matched replacement #Delta p_{x};p_{x}^{after}-p_{x}^{before} (GeV/c);Tracks", 200, -2., 2.);
   auto *h_dpy = new TH1D("h_deltaPy", "Matched replacement #Delta p_{y};p_{y}^{after}-p_{y}^{before} (GeV/c);Tracks", 200, -2., 2.);
   auto *h_dr = new TH1D("h_deltaR_ep", "Displacement in (#eta,#phi);#sqrt{(#Delta#eta)^{2}+(#Delta#phi)^{2}};Tracks", 200, 0., 7.);

   auto *h_step = new TH1D("h_levyStep", "L\xC3\xA9vy-walk step size in (#eta,#phi);step size (arb.);Samples", 200, 0., 1.5);
   auto *h_pool = new TH1D("h_poolSize", "CMC candidate pool size per pass;pool size;Passes", 400, 0., 4000.);
   auto *h_unm = new TH1D("h_unmatchedAfterPass", "Unmatched targets after pass;N_{unmatched};Passes", 150, 0., 150.);

   auto *h2_pxy_bf = new TH2D("h2_px_py_before", "Accepted tracks before injection;p_{x} (GeV/c);p_{y} (GeV/c)", 160, -2.2, 2.2, 160, -2.2, 2.2);
   auto *h2_pxy_af = new TH2D("h2_px_py_after", "Accepted tracks after injection;p_{x} (GeV/c);p_{y} (GeV/c)", 160, -2.2, 2.2, 160, -2.2, 2.2);
   auto *h2_pxy_rep = new TH2D("h2_px_py_replaced", "Injected replacement tracks after injection;p_{x} (GeV/c);p_{y} (GeV/c)", 160, -2.2, 2.2, 160, -2.2, 2.2);
   auto *h2_ep_bf = new TH2D("h2_eta_phi_before", "Accepted tracks before injection;#eta;#phi", 100, -1., 1., 128, 0., 2. * TMath::Pi());
   auto *h2_ep_af = new TH2D("h2_eta_phi_after", "Accepted tracks after injection;#eta;#phi", 100, -1., 1., 128, 0., 2. * TMath::Pi());

   auto *h_mult = new TH1D("h_multiplicity", "Accepted multiplicity per event;N_{accepted};Events", 2000, 0., 4000.);
   auto *h_in_mult = new TH1D("h_in_multiplicity", "Injected multiplicity per event;N_{Injected};Events", 2000, 0., 4000.);
   auto *h_av = new TH1D("h_av_bin_content", "Average bin content;M;<n>", _M_bins, -0.5, _M_bins - 0.5);
   auto *h_fq2 = new TH1D("h_fq2", "Factorial moment F_{2};M^2;F_{2}", _M_bins, -0.5, _M_bins - 0.5);
   auto *h_fq3 = new TH1D("h_fq3", "Factorial moment F_{3};M^2;F_{3}", _M_bins, -0.5, _M_bins - 0.5);
   auto *h_fq4 = new TH1D("h_fq4", "Factorial moment F_{4};M^2;F_{4}", _M_bins, -0.5, _M_bins - 0.5);
   auto *h_fq5 = new TH1D("h_fq5", "Factorial moment F_{5};M^2;F_{5}", _M_bins, -0.5, _M_bins - 0.5);
   setMLabel(h_av);
   setMLabel(h_fq2);
   setMLabel(h_fq3);
   setMLabel(h_fq4);
   setMLabel(h_fq5);

   auto *h_in_av = new TH1D("h_in_av_bin_content", "Average bin content;M;<n>", _M_bins, -0.5, _M_bins - 0.5);
   auto *h_in_fq2 = new TH1D("h_in_fq2", "Factorial moment F_{2};M^2;F_{2}", _M_bins, -0.5, _M_bins - 0.5);
   auto *h_in_fq3 = new TH1D("h_in_fq3", "Factorial moment F_{3};M^2;F_{3}", _M_bins, -0.5, _M_bins - 0.5);
   auto *h_in_fq4 = new TH1D("h_in_fq4", "Factorial moment F_{4};M^2;F_{4}", _M_bins, -0.5, _M_bins - 0.5);
   auto *h_in_fq5 = new TH1D("h_in_fq5", "Factorial moment F_{5};M^2;F_{5}", _M_bins, -0.5, _M_bins - 0.5);
   setMLabel(h_in_av);
   setMLabel(h_in_fq2);
   setMLabel(h_in_fq3);
   setMLabel(h_in_fq4);
   setMLabel(h_in_fq5);

   const std::array<TH1D *, _fq_ord> _hFq = {h_fq2, h_fq3, h_fq4, h_fq5};
   const std::array<TH1D *, _fq_ord> _hInFq = {h_in_fq2, h_in_fq3, h_in_fq4, h_in_fq5};

   // Subsample accumulators for error estimation
   SubsampleAccum _subBf;   // before injection
   SubsampleAccum _subInj;  // after injection

   ///////////////////////SPARSE/////////////////////////
   const Int_t _sp_bins[4] = {100, 128, 200, static_cast<Int_t>(_nev > 0 ? _nev : 1)};
   const Double_t _sp_min[4] = {-_eta_max, 0., 0., 0.};
   const Double_t _sp_max[4] = {_eta_max, 2. * TMath::Pi(), 20., static_cast<Double_t>(_nev > 0 ? _nev : 1)};
   auto *h_sp = new THnSparseD("hSparseEtaPhiBimEvent", "Accepted tracks;#eta;#phi;b_{im};event", 4, _sp_bins, _sp_min, _sp_max);

   std::array<TH2D *, _M_bins> h2_ep_tmp{};
   std::array<TH2D *, _M_bins> h2_ep_in_tmp{};
   for (Int_t _M = 0; _M < _M_bins; ++_M) {
      const auto _mb = getMVal(_M);
      h2_ep_tmp[_M] = new TH2D(Form("h2_ep_tmp_%d", _M), Form("h2_ep_tmp_%d", _M), _mb, -_eta_max, _eta_max, _mb, 0., 2. * TMath::Pi());
      h2_ep_tmp[_M]->SetDirectory(nullptr);
      h2_ep_in_tmp[_M] = new TH2D(Form("h2_ep_in_tmp_%d", _M), Form("h2_ep_in_tmp_%d", _M), _mb, -_eta_max, _eta_max, _mb, 0., 2. * TMath::Pi());
      h2_ep_in_tmp[_M]->SetDirectory(nullptr);
   }

   TRandom3 _rng(static_cast<UInt_t>(0x5EEDC0FFEEULL));

   Long64_t _tot_acc = 0;
   Long64_t _tot_req = 0;
   Long64_t _tot_inj = 0;
   Long64_t _nev_inj = 0;
   std::cout << "Processing " << _nev << " entries with lambda=" << _lam << " ...\n";

   //////////////////EVENT LOOP//////////////////////////
   for (Long64_t _ev = 0; _ev < _nev; ++_ev) {
      const auto _nb = _tr->GetEntry(_ev);
      if (_nb <= 0) {
         std::cerr << "Warning: failed GetEntry(" << _ev << ")\n";
         continue;
      }
      if (np < 0 || np >= _np_max) {
         std::cerr << "Warning: entry " << _ev << " has invalid np=" << np << " (buffer max " << _np_max << ")\n";
         continue;
      }

      if (bim > 3.5) continue;

      std::vector<AccTrk> _acc;
      _acc.reserve(np);
      for (Int_t _M = 0; _M < _M_bins; ++_M) {
         h2_ep_tmp[_M]->Reset();
         h2_ep_in_tmp[_M]->Reset();
      }
      //////////////TRACK LOOP//////////////////////////
      for (Int_t _trk = 0; _trk < np; ++_trk) {
         const auto _px = static_cast<Double_t>(px[_trk]);
         const auto _py = static_cast<Double_t>(py[_trk]);
         const auto _pz = static_cast<Double_t>(pz[_trk]);
         const auto _pid = id[_trk];
         const auto _st = ist[_trk];

         if (!passCut(_px, _py, _pz, _pid, _st)) continue;

         const auto _pt = getPt(_px, _py);
         const auto _eta = getEta(_pt, _pz);
         const auto _phi = getPhi(_px, _py);
         _acc.push_back({_trk, _pt, _px, _py, _pz});

         h_pt_bf->Fill(_pt);
         h_eta_bf->Fill(_eta);
         h_phi_bf->Fill(_phi);
         h2_pxy_bf->Fill(_px, _py);
         h2_ep_bf->Fill(_eta, _phi);

         for (Int_t _M = 0; _M < _M_bins; ++_M) h2_ep_tmp[_M]->Fill(_eta, _phi);

         const Double_t _sp_val[4] = {_eta, _phi, static_cast<Double_t>(bim), static_cast<Double_t>(_ev)};
         h_sp->Fill(_sp_val);
      }

      /////////////////FILL FQ/////////////////////////
      {
         auto _fqEv = calcFqEvent(h2_ep_tmp);
         _subBf.addEvent(_fqEv);
      }

      _tot_acc += _acc.size();
      h_nacc->Fill(static_cast<Double_t>(_acc.size()));
      h_mult->Fill(static_cast<Double_t>(_acc.size()));

      /////////////////INJECTION/////////////////////////
      const auto _nrep = getNRep(static_cast<Int_t>(_acc.size()), _rng);
      _tot_req += _nrep;
      h_nreq->Fill(_nrep);

      Int_t _ninj = 0;
      if (!_acc.empty() && _nrep > 0) {
         // Seed: pick one accepted track and use its (eta,phi) as the walk starting point.
         const auto _seed_i = static_cast<std::size_t>(_rng.Integer(_acc.size()));
         const auto &_seed = _acc[_seed_i];
         const auto _seed_eta = getEta(_seed.pt, _seed.pz);
         const auto _seed_phi = getPhi(_seed.px, _seed.py);

         // Pick which tracks to replace (random subset of _nrep from accepted).
         auto _tgt = _acc;
         shuf(_tgt, _rng);
         if (static_cast<Int_t>(_tgt.size()) > _nrep) _tgt.resize(_nrep);

         // Generate a single pool of (eta,phi) CMC candidates.
         // Pool is oversized relative to targets so it rarely runs short.
         const auto _psz = TMath::Max(_pool_min, _cand_mul * static_cast<Int_t>(_tgt.size()));
         h_pool->Fill(_psz);
         const auto _pool = makePool(_seed_eta, _seed_phi, _psz, _rng, h_step);
         const Int_t _navail = static_cast<Int_t>(_pool.size());

         // Record how many targets could not be served (pool ran short).
         h_unm->Fill(TMath::Max(0, static_cast<Int_t>(_tgt.size()) - _navail));

         // Sequential assignment: target[i] gets pool[i].
         // pT is preserved EXACTLY by construction — no tolerance matching needed.
         for (Int_t _ti = 0; _ti < static_cast<Int_t>(_tgt.size()) && _ti < _navail; ++_ti) {
            const auto &_t = _tgt[_ti];
            const auto &_cand = _pool[_ti];

            const auto _px_old = static_cast<Double_t>(px[_t.idx]);
            const auto _py_old = static_cast<Double_t>(py[_t.idx]);
            const auto _pt_orig = _t.pt; // preserved exactly
            const auto _eta_old = getEta(_pt_orig, _t.pz);
            const auto _phi_old = getPhi(_px_old, _py_old);

            // Reconstruct full 3-momentum from (eta_cmc, phi_cmc, pT_orig).
            const auto _px_new = _pt_orig * TMath::Cos(_cand.phi);
            const auto _py_new = _pt_orig * TMath::Sin(_cand.phi);
            const auto _pz_new = _pt_orig * TMath::SinH(_cand.eta);

            px[_t.idx] = static_cast<Float_t>(_px_new);
            py[_t.idx] = static_cast<Float_t>(_py_new);
            pz[_t.idx] = static_cast<Float_t>(_pz_new); // pz must be updated too
            ++_ninj;

            // QA
            const auto _deta = _cand.eta - _eta_old;
            const auto _dphi = _cand.phi - _phi_old;
            const auto _dr = TMath::Sqrt(_deta * _deta + _dphi * _dphi);

            h_rpt_bf->Fill(_pt_orig);
            h_rpt_af->Fill(_pt_orig); // pT unchanged by design
            h_reta_bf->Fill(_eta_old);
            h_reta_af->Fill(_cand.eta);
            h_rphi_bf->Fill(_phi_old);
            h_rphi_af->Fill(_cand.phi);
            h_dpt->Fill(0.); // pT exactly preserved
            h_adpt->Fill(0.);
            h_dpx->Fill(_px_new - _px_old);
            h_dpy->Fill(_py_new - _py_old);
            h_dr->Fill(_dr);
            h2_pxy_rep->Fill(_px_new, _py_new);
         }
      }

      _tot_inj += _ninj;
      if (_ninj > 0) ++_nev_inj;
      h_ninj->Fill(_ninj);
      if (!_acc.empty()) {
         h_lam_e->Fill(static_cast<Double_t>(_ninj) / static_cast<Double_t>(_acc.size()));
      }

      for (const auto &_t : _acc) {
         const auto _px = static_cast<Double_t>(px[_t.idx]);
         const auto _py = static_cast<Double_t>(py[_t.idx]);
         const auto _pz = static_cast<Double_t>(pz[_t.idx]);
         const auto _pt = getPt(_px, _py);
         const auto _eta = getEta(_pt, _pz);
         const auto _phi = getPhi(_px, _py);
         if (_pt < _pt_min || _pt > _pt_max || TMath::Abs(_eta) >= _eta_max) continue;

         h_pt_af->Fill(_pt);
         h_eta_af->Fill(_eta);
         h_phi_af->Fill(_phi);
         h2_pxy_af->Fill(_px, _py);
         h2_ep_af->Fill(_eta, _phi);

         for (Int_t _M = 0; _M < _M_bins; ++_M) h2_ep_in_tmp[_M]->Fill(_eta, _phi);
      }

      h_in_mult->Fill(_acc.size());
      {
         auto _fqEvInj = calcFqEvent(h2_ep_in_tmp);
         _subInj.addEvent(_fqEvInj);
      }

      // Close subsamples at boundary
      if (_subBf.nEvInSub >= _sub_size) _subBf.closeSubsample();
      if (_subInj.nEvInSub >= _sub_size) _subInj.closeSubsample();
      _tout->Fill();

      if ((_ev + 1) % _prog_n == 0 || _ev + 1 == _nev) {
         std::cout << "Processed " << (_ev + 1) << "/" << _nev << " entries" << ", accepted tracks so far=" << _tot_acc << ", requested replacements=" << _tot_req << ", injected=" << _tot_inj << "\n";
      }
   }

   ////////////////END EVENT LOOP//////////////////////////
   // Close any remaining partial subsamples
   _subBf.closeSubsample();
   _subInj.closeSubsample();

   const Int_t _nSubBf = static_cast<Int_t>(_subBf.subFq[0][0].size());
   const Int_t _nSubInj = static_cast<Int_t>(_subInj.subFq[0][0].size());
   std::cout << "Finished processing " << _nev << " entries" << ", accepted tracks=" << _tot_acc << ", requested replacements=" << _tot_req << ", injected=" << _tot_inj << "\n";
   std::cout << "Subsampling: " << _nSubBf << " subsamples (before), " << _nSubInj << " subsamples (injected), _sub_size=" << _sub_size << "\n";

   ///////////////WRITING OUTPUT/////////////////////////
   _fout->cd();
   _tout->Write();
   _fout->Close();

   _fqa->cd();
   std::cout << "Writing QA histograms to " << _qa_name << "\n";

   ////////////////COMPUTE FQ WITH ERRORS FROM SUBSAMPLING////////////////////////
   // Before injection: mean ± stderr from subsamples → h_fq* bins + h_av
   _subBf.finalize(_hFq, h_av);
   // After injection: mean ± stderr from subsamples → h_in_fq* bins + h_in_av
   _subInj.finalize(_hInFq, h_in_av);

   ////////////////WRITE HISTOGRAMS/////////////////////////
   h_nacc->Write();
   h_nreq->Write();
   h_ninj->Write();
   h_lam_e->Write();
   h_pt_bf->Write();
   h_pt_af->Write();
   h_eta_bf->Write();
   h_eta_af->Write();
   h_phi_bf->Write();
   h_phi_af->Write();
   h_rpt_bf->Write();
   h_rpt_af->Write();
   h_reta_bf->Write();
   h_reta_af->Write();
   h_rphi_bf->Write();
   h_rphi_af->Write();
   h_dpt->Write();
   h_adpt->Write();
   h_dpx->Write();
   h_dpy->Write();
   h_dr->Write();
   h_step->Write();
   h_pool->Write();
   h_unm->Write();
   h2_pxy_bf->Write();
   h2_pxy_af->Write();
   h2_pxy_rep->Write();
   h2_ep_bf->Write();
   h2_ep_af->Write();
   h_mult->Write();
   h_av->Write();
   h_fq2->Write();
   h_fq3->Write();
   h_fq4->Write();
   h_fq5->Write();
   h_in_av->Write();
   h_in_fq2->Write();
   h_in_fq3->Write();
   h_in_fq4->Write();
   h_in_fq5->Write();
   h_sp->Write();
   _fqa->Close();
   _fin->Close();

   std::cout << "CMC injection finished.\n"
             << "  Input file:  " << _in_name << "\n"
             << "  Output file: " << _out_name << "\n"
             << "  QA file:     " << _qa_name << "\n"
             << "  Subsamples (before/inj):     " << _nSubBf << " / " << _nSubInj << " (size=" << _sub_size << ")\n"
             << "  Events with injected tracks: " << _nev_inj << " / " << _nev << "\n"
             << "  Total accepted tracks:       " << _tot_acc << "\n"
             << "  Requested replacements:      " << _tot_req << "\n"
             << "  Successful injections:       " << _tot_inj << "\n";
}
