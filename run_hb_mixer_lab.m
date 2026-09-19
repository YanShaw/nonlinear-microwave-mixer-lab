function results = run_hb_mixer_lab(showFigures)
% RUN_HB_MIXER_LAB  Chapter 3: HB -> conversion matrix -> time-domain check.
% No add-on toolboxes. Run: results = run_hb_mixer_lab;
% All voltages are V, currents A, frequencies Hz, capacitances F internally.
% See README.md for the circuit, conventions, and guided experiments.
if nargin == 0, showFigures = true; end
out = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(out, 'dir'), mkdir(out); end
p.Rs = 50; p.RL = 1000;
p.Is = 1e-6; p.nVT = 0.026;
p.Cj0 = 0.25e-12; p.Cpar = 0.15e-12;
p.Vj = 0.7; p.Fc = 0.5;
p.Vdc = 0.05; p.Vlo = 0.50; p.Vrf = 1e-3;
p.flo = 1e9; p.frf = 1.1e9; p.fif = p.frf-p.flo;
p.K = 47; p.N = 17; p.M = 2048;
fprintf('Chapter 3 mixer lab | MATLAB %s\n', version);
fprintf('LO %.3f GHz, RF %.3f GHz, IF %.1f MHz\n', ...
    p.flo/1e9, p.frf/1e9, p.fif/1e6);

% 1. Solve the nonlinear LO-only steady state by real Fourier-basis HB.
hb = solve_hb(p, p.K, p.M);
cm = conversion(p, hb, p.N);
fprintf('HB residual: %.3g A; IF voltage: %.6g V peak\n', ...
    hb.residual, abs(cm.Vif));

% 2. Independent integration of the ORIGINAL nonlinear circuit equation.
td0 = transient(p, 0, 1e-8);
td = transient(p, p.Vrf, 1e-8);
td.Vif = tone(td.v, td.u, p.fif/p.flo);
vHB = basis(2*pi*td0.u, p.K)*hb.x;
dvTD = td.v-td0.v;
dvCM = real(exp(1i*2*pi*td.u*cm.freq.'/p.flo)*cm.V);

% 3. Harmonic truncation: converge HB AND the IF prediction, not just F=0.
Ks = [3 5 9 15 23 31 47];
hbResidual = zeros(size(Ks)); ifPeak = hbResidual; loPeak = hbResidual;
for k = 1:numel(Ks)
    h = solve_hb(p, Ks(k), p.M);
    c = conversion(p, h, p.N);
    hbResidual(k) = h.residual;
    ifPeak(k) = abs(c.Vif);
    loPeak(k) = hypot(h.x(2), h.x(Ks(k)+2));
end
harmonicTable = table(Ks(:), hbResidual(:), loPeak(:), ifPeak(:), ...
    'VariableNames', {'K','HB_residual_A','LO_node_peak_V','IF_peak_V'});

% 4. Sideband truncation is separate from the LO harmonic count.
Ns = [1 2 3 5 9 13 17 23]; sidebandIF = zeros(size(Ns));
for k = 1:numel(Ns)
    c = conversion(p, hb, Ns(k)); sidebandIF(k) = abs(c.Vif);
end
sidebandTable = table(Ns(:), sidebandIF(:), ...
    'VariableNames', {'N','IF_peak_V'});

% 5. Increasing RF amplitude tests the small-signal approximation itself.
rf = [0.1 1 10 50 150 300]*1e-3;
predicted = abs(cm.Vif)*rf/p.Vrf;
measured = zeros(size(rf)); rfComplexError = measured;
for k = 1:numel(rf)
    if rf(k) == p.Vrf, tr = td; else, tr = transient(p, rf(k), 1e-8); end
    z = tone(tr.v, tr.u, p.fif/p.flo);
    measured(k) = abs(z);
    rfComplexError(k) = abs(z-cm.Vif*rf(k)/p.Vrf)/abs(cm.Vif*rf(k)/p.Vrf);
end
rfTable = table(rf(:), predicted(:), measured(:), rfComplexError(:), ...
    'VariableNames', {'RF_source_peak_V','CM_IF_peak_V','TD_IF_peak_V','Complex_relative_error'});

% Checks target independent mathematics/physics, with tolerances visible.
checks.HB_residual_A = hb.residual;
checks.LO_waveform_relative_rms = norm(vHB-td0.v)/norm(td0.v);
checks.IF_complex_relative_error = abs(td.Vif-cm.Vif)/abs(cm.Vif);
checks.small_signal_waveform_relative_rms = norm(dvTD-dvCM)/norm(dvCM);
checks.K31_to_K47_IF_relative_change = abs(ifPeak(end)-ifPeak(end-1))/ifPeak(end);
checks.N17_to_N23_IF_relative_change = abs(sidebandIF(end)-sidebandIF(end-1))/sidebandIF(end);
hFine = solve_hb(p, p.K, 2*p.M); cFine = conversion(p, hFine, p.N);
checks.M2048_to_M4096_IF_relative_change = abs(cFine.Vif-cm.Vif)/abs(cFine.Vif);
tFine = transient(p, p.Vrf, 1e-10);
checks.TD_tolerance_IF_relative_change = abs(tone(tFine.v,tFine.u,p.fif/p.flo)-td.Vif)/abs(td.Vif);
pOff = p; pOff.Vlo = 0;
hOff = solve_hb(pOff, p.K, p.M); cOff = conversion(pOff, hOff, p.N);
checks.no_LO_IF_peak_V = abs(cOff.Vif);
checks.jacobian_directional_relative_error = jacobian_check(p, hb);
assert(checks.HB_residual_A < 1e-9, 'HB did not converge.');
assert(checks.LO_waveform_relative_rms < 2e-3, 'HB disagrees with LO-only ODE.');
assert(checks.IF_complex_relative_error < 0.02, 'Conversion matrix disagrees with weak-RF ODE.');
assert(checks.small_signal_waveform_relative_rms < 0.01, 'Incremental waveform mismatch.');
assert(checks.K31_to_K47_IF_relative_change < 0.005, 'Increase K.');
assert(checks.N17_to_N23_IF_relative_change < 0.005, 'Increase N.');
assert(checks.M2048_to_M4096_IF_relative_change < 1e-4, 'Increase sampling M.');
assert(checks.TD_tolerance_IF_relative_change < 1e-3, 'Tighten ODE tolerance.');
assert(checks.no_LO_IF_peak_V < 1e-10, 'No-LO circuit should not mix at first order.');
assert(checks.jacobian_directional_relative_error < 1e-5, 'HB Jacobian check failed.');
assert(rfComplexError(end) > 0.02, 'RF sweep did not yet show small-signal breakdown.');

results.parameters = p; results.hb = hb; results.cm = cm;
results.td = td; results.tdLO = td0; results.checks = checks;
results.harmonicTable = harmonicTable;
results.sidebandTable = sidebandTable; results.rfTable = rfTable;
results.IF_load_power_W = abs(cm.Vif)^2/(2*p.RL);
results.RF_available_power_W = p.Vrf^2/(8*p.Rs);
results.conversion_gain_dB = 10*log10(results.IF_load_power_W/results.RF_available_power_W);
writetable(harmonicTable, fullfile(out,'harmonics.csv'));
writetable(sidebandTable, fullfile(out,'sidebands.csv'));
writetable(rfTable, fullfile(out,'rf_amplitude.csv'));
save(fullfile(out,'lab_results.mat'), 'results');
fid = fopen(fullfile(out,'verification.txt'),'w');
fprintf(fid, 'MATLAB: %s\n',version);
names = fieldnames(checks);
for k = 1:numel(names), fprintf(fid,'%s = %.12g\n',names{k},checks.(names{k})); end
fprintf(fid,'IF_peak_V = %.12g\nIF_phase_deg = %.12g\nconversion_gain_dB = %.12g\n', ...
    abs(cm.Vif), angle(cm.Vif)*180/pi, results.conversion_gain_dB);
fprintf(fid,'All stated verification thresholds passed.\n'); fclose(fid);
draw_results(p, hb, cm, td0, td, vHB, dvTD, dvCM, results, showFigures, out);
disp(harmonicTable); disp(rfTable); disp(checks);
fprintf('All checks passed. Results: %s\n', out);
end

function B = basis(theta,K)
theta = theta(:);
B = [ones(size(theta)), cos(theta*(1:K)), sin(theta*(1:K))];
end

function [id,gd,q,c] = diode(v,p)
% Teaching Schottky-like model; not a fitted commercial-device model.
ev = exp(v/p.nVT); id = p.Is*expm1(v/p.nVT); gd = p.Is/p.nVT*ev;
% Abrupt-junction charge, continued above Fc*Vj with continuous C and dC/dV.
vf = p.Fc*p.Vj; low = v <= vf;
q = zeros(size(v)); c = q;
q(low) = 2*p.Cj0*p.Vj*(1-sqrt(1-v(low)/p.Vj));
c(low) = p.Cj0./sqrt(1-v(low)/p.Vj);
cf = p.Cj0/sqrt(1-p.Fc);
dcf = p.Cj0/(2*p.Vj)*(1-p.Fc)^(-1.5);
qf = 2*p.Cj0*p.Vj*(1-sqrt(1-p.Fc));
d = v(~low)-vf;
q(~low) = qf+cf*d+0.5*dcf*d.^2; c(~low) = cf+dcf*d;
q = q+p.Cpar*v; c = c+p.Cpar;
end

function h = solve_hb(p,K,M)
theta = (0:M-1)'*2*pi/M;
B = basis(theta,K); P = B'/M; P(2:end,:) = 2*P(2:end,:);
D = zeros(2*K+1);
for k = 1:K, D(1+k,1+K+k)=k; D(1+K+k,1+k)=-k; end
x = zeros(2*K+1,1); x(1) = p.Vdc/(1+p.Rs/p.RL);
source = zeros(size(x)); source(1) = p.Vdc;
history = []; stage = [];
% Source continuation helps Newton reach the physically relevant pumped state.
for drive = linspace(0,p.Vlo,11)
    source(2) = drive;
    for it = 1:70
        [F,J] = hb_residual(x,p,B,P,D,source);
        err = norm(F,inf); history(end+1,1)=err; stage(end+1,1)=drive; %#ok<AGROW>
        if err < 1e-11, break; end
        dx = -J\F; alpha = 1; accepted = false;
        for ls = 1:25
            xn = x+alpha*dx;
            Fn = hb_residual(xn,p,B,P,D,source);
            if all(isfinite(Fn)) && norm(Fn,inf) < err
                x = xn; accepted = true; break;
            end
            alpha = alpha/2;
        end
        assert(accepted, 'Newton line search failed; inspect parameters.');
    end
    assert(err < 1e-11, 'Newton iteration limit reached.');
end
[id,gd,q,c] = diode(B*x,p);
h.x=x; h.K=K; h.M=M; h.theta=theta; h.v=B*x;
h.id=id; h.g=gd; h.q=q; h.c=c;
h.history=history; h.driveHistory=stage;
h.residual=norm(hb_residual(x,p,B,P,D,source),inf);
end

function [F,J] = hb_residual(x,p,B,P,D,source)
[id,gd,q,c] = diode(B*x,p);
a = 1/p.Rs+1/p.RL;
F = a*x+P*id+2*pi*p.flo*D*(P*q)-source/p.Rs;
if nargout > 1
    J = a*eye(numel(x))+P*(gd.*B)+2*pi*p.flo*D*(P*(c.*B));
end
end

function cm = conversion(p,h,N)
% Peak phasors on ONE signed-frequency family: f_n=f_IF+n*f_LO.
% Real incremental waveform = real(sum_n V_n*exp(j*2*pi*f_n*t)).
n = (-N:N)'; ell = -2*N:2*N;
E = exp(-1i*h.theta*ell);
G = (h.g.'*E)/h.M; C = (h.c.'*E)/h.M;
index = n-n.'+2*N+1;
Gmat = reshape(G(index),size(index)); Cmat = reshape(C(index),size(index));
freq = p.fif+n*p.flo;
Y = (1/p.Rs+1/p.RL)*eye(numel(n))+Gmat+1i*diag(2*pi*freq)*Cmat;
rhs = zeros(numel(n),1); rhs(n==1)=p.Vrf/p.Rs;
V = Y\rhs;
cm.n=n; cm.freq=freq; cm.G=Gmat; cm.C=Cmat; cm.Y=Y;
cm.V=V; cm.Vif=V(n==0); cm.rhs=rhs;
end

function tr = transient(p,vrf,rtol)
% u = f_LO*t (number of LO cycles). Not a circuit approximation.
rhs = @(u,v) ode_rhs(u,v,p,vrf);
opt = odeset('RelTol',rtol,'AbsTol',rtol*0.01,'MaxStep',1/40);
% Frequencies have ratio 11/10: the final 40 LO cycles contain 4 IF periods.
sol = ode15s(rhs,[0 80],0,opt);
tr.u = 40+(0:8191)'*40/8192;
tr.v = deval(sol,tr.u).';
tr.settling_relative = norm(deval(sol,tr.u)-deval(sol,tr.u-10))/norm(tr.v);
assert(tr.settling_relative < 1e-5,'Transient has not reached periodic steady state.');
end

function dv = ode_rhs(u,v,p,vrf)
[id,~,~,c] = diode(v,p);
vs = p.Vdc+p.Vlo*cos(2*pi*u)+vrf*cos(2*pi*p.frf/p.flo*u);
dv = ((vs-v)/p.Rs-v/p.RL-id)/(p.flo*c);
end

function z = tone(v,u,ratio)
% Peak phasor at a POSITIVE frequency; coherent, endpoint-excluded sampling.
z = 2*mean(v.*exp(-1i*2*pi*ratio*u));
end

function e = jacobian_check(p,h)
K=h.K; B=basis(h.theta,K); P=B'/h.M; P(2:end,:)=2*P(2:end,:);
D=zeros(2*K+1);
for k=1:K, D(1+k,1+K+k)=k; D(1+K+k,1+k)=-k; end
s=zeros(size(h.x)); s(1)=p.Vdc; s(2)=p.Vlo;
[~,J]=hb_residual(h.x,p,B,P,D,s);
d=sin((1:numel(h.x))'); d=d/norm(d); epsv=1e-6;
fd=(hb_residual(h.x+epsv*d,p,B,P,D,s)-hb_residual(h.x-epsv*d,p,B,P,D,s))/(2*epsv);
e=norm(fd-J*d)/norm(J*d);
end

function draw_results(p,h,cm,t0,t,vhb,dvtd,dvcm,r,visible,out)
if visible, vis='on'; else, vis='off'; end
f=figure('Name','1 - LO harmonic balance','Visible',vis,'Position',[60 60 1100 760]);
tiledlayout(2,2);
nexttile; plot(h.theta/(2*pi),p.Vdc+p.Vlo*cos(h.theta),'--',h.theta/(2*pi),h.v,'LineWidth',1.3);
xlabel('LO cycles'); ylabel('Voltage (V)'); legend('Source','Diode node'); grid on; title('Same circuit, different voltages');
nexttile; plot(h.theta/(2*pi),h.id*1e3,'LineWidth',1.3); xlabel('LO cycles'); ylabel('Diode current (mA)'); grid on; title('Pulsed nonlinear current');
nexttile; amp=[abs(h.x(1)); hypot(h.x(2:h.K+1),h.x(h.K+2:end))];
stem(0:h.K,amp*1e3,'filled'); xlabel('Harmonic order k (0 = DC)'); ylabel('Node voltage peak (mV)'); grid on; title('HB voltage spectrum');
nexttile; ix=find(abs(h.driveHistory-p.Vlo)<1e-14); semilogy(0:numel(ix)-1,h.history(ix),'o-');
xlabel('Newton iteration at final LO amplitude'); ylabel('Maximum KCL residual (A)'); grid on; title('Convergence of retained equations');
finish(f,out,'01_harmonic_balance',visible);

f=figure('Name','2 - Conversion matrices','Visible',vis,'Position',[80 80 1100 760]); tiledlayout(2,2);
nexttile; plot(h.theta/(2*pi),h.g*1e3,'LineWidth',1.3); xlabel('LO cycles'); ylabel('g(t) (mS)'); grid on; title('Differential conductance at pumped state');
nexttile; plot(h.theta/(2*pi),h.c*1e12,'LineWidth',1.3); xlabel('LO cycles'); ylabel('c(t), including Cpar (pF)'); grid on; title('Differential charge capacitance');
nexttile; imagesc(cm.n,cm.n,abs(cm.G)*1e3); axis xy; colorbar; xlabel('Input frequency index m'); ylabel('Output frequency index n'); title('|G(n,m)| in mS: frequency coupling');
nexttile; stem(cm.n,abs(cm.V)*1e3,'filled'); xlabel('n: f_n = 0.1 GHz + n * 1 GHz'); ylabel('Small-signal peak magnitude (mV)'); grid on; title('RF source at n=1; IF output at n=0');
finish(f,out,'02_conversion_matrix',visible);

f=figure('Name','3 - Independent time-domain validation','Visible',vis,'Position',[100 100 1100 760]); tiledlayout(2,2);
nexttile; ix=t0.u<42; plot(t0.u(ix)-40,t0.v(ix),t0.u(ix)-40,vhb(ix),'--','LineWidth',1.2);
xlabel('LO cycles after settling'); ylabel('Voltage (V)'); legend('Original ODE, LO only','HB'); grid on; title('LO waveform: two independent solvers');
nexttile; ix=t.u<50; plot(t.u(ix)-40,dvtd(ix)*1e3,t.u(ix)-40,dvcm(ix)*1e3,'--','LineWidth',1.1);
xlabel('LO cycles after settling'); ylabel('Incremental voltage (mV)'); legend('ODE: LO+RF minus LO','Conversion matrix'); grid on; title('Small-signal waveform');
nexttile; fp=[.1 .9 1.1 1.9 2.1]*p.flo; av=zeros(size(fp)); bv=av;
for k=1:numel(fp)
    av(k)=abs(tone(dvtd,t.u,fp(k)/p.flo));
    j=find(abs(abs(cm.freq)-fp(k))<1); bv(k)=abs(cm.V(j));
end
bar(fp/1e9,[av(:),bv(:)]*1e6); xlabel('Physical positive frequency (GHz)'); ylabel('Peak voltage (uV)'); legend('Nonlinear ODE','Conversion matrix'); grid on; title('Unequal mixing-line amplitudes');
nexttile; tt=(0:500)'/500/p.fif;
plot(tt*1e9,real(t.Vif*exp(1i*2*pi*p.fif*tt))*1e6,tt*1e9,real(cm.Vif*exp(1i*2*pi*p.fif*tt))*1e6,'--','LineWidth',1.3);
xlabel('Time (ns)'); ylabel('IF voltage (uV)'); legend('ODE IF component','Conversion-matrix IF'); grid on; title('IF component: extracted 100 MHz');
finish(f,out,'03_time_domain_check',visible);

f=figure('Name','4 - Approximation limits','Visible',vis,'Position',[120 120 1100 760]); tiledlayout(2,2);
nexttile; plot(r.harmonicTable.K,r.harmonicTable.IF_peak_V*1e6,'o-'); xlabel('HB harmonic cutoff K'); ylabel('Predicted IF peak (uV)'); grid on; title('Harmonic truncation changes the answer');
nexttile; plot(r.sidebandTable.N,r.sidebandTable.IF_peak_V*1e6,'o-'); xlabel('Conversion sideband cutoff N'); ylabel('Predicted IF peak (uV)'); grid on; title('Separate sideband-convergence test');
nexttile; loglog(r.rfTable.RF_source_peak_V*1e3,r.rfTable.CM_IF_peak_V*1e6,'o-',r.rfTable.RF_source_peak_V*1e3,r.rfTable.TD_IF_peak_V*1e6,'s-');
xlabel('RF source peak (mV)'); ylabel('IF peak (uV)'); legend('Linearized prediction','Full nonlinear ODE'); grid on; title('When RF is no longer small');
nexttile; semilogx(r.rfTable.RF_source_peak_V*1e3,r.rfTable.Complex_relative_error*100,'o-'); xlabel('RF source peak (mV)'); ylabel('Complex IF error (%)'); grid on; title('Amplitude AND phase error');
finish(f,out,'04_convergence_and_limits',visible);
end

function finish(f,out,name,visible)
f.Color='white';
set(findall(f,'Type','axes'),'FontSize',10,'TitleFontSizeMultiplier',1.1, ...
    'Color','white','XColor','black','YColor','black','GridColor',[0.3 0.3 0.3]);
set(findall(f,'Type','text'),'Color','black');
set(findall(f,'Type','legend'),'Color','white','TextColor','black','EdgeColor',[0.3 0.3 0.3]);
set(findall(f,'Type','colorbar'),'Color','black');
layouts=findall(f,'Type','tiledlayout');
for k=1:numel(layouts), layouts(k).Padding='loose'; layouts(k).TileSpacing='loose'; end
drawnow;
exportgraphics(f,fullfile(out,[name '.png']),'Resolution',150,'BackgroundColor','white');
savefig(f,fullfile(out,[name '.fig']));
if ~visible, close(f); end
end
