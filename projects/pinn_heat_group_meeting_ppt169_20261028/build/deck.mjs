import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const W = 960, H = 5184775 / 9525;
const C = { red:'#A51E36', gray:'#555759', sub:'#6C6E70', pale:'#C98C98', line:'#D8D8D8', white:'#FFFFFF' };
const FONT = 'DengXian';

function text(s, name, value, x, y, w, h, size=23, color=C.gray, bold=false, font=FONT, align='left') {
  const a=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  a.text=value;
  a.text.style={typeface:font,fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',insets:0,wrap:'none'};
  a.text.insets=0;
  return a;
}
function base(p, n, title) {
  const s=p.slides.add({width:W,height:H,background:{fill:C.white}});
  text(s,'page-title',title,80,44,810,48,32,C.red);
  text(s,'page-number',String(n).padStart(2,'0'),858,510,30,18,13,C.sub,false,FONT,'right');
  return s;
}
async function image(s,name,file,x,y,w,h){
  const b=await fs.readFile(path.join(root,'images',file));
  const mime=file.endsWith('.jpg')?'image/jpeg':'image/png';
  s.images.add({name,dataUrl:'data:'+mime+';base64,'+b.toString('base64'),contentType:mime,alt:name,fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
function arrow(s,x,y,w){s.shapes.add({name:'flow-arrow',geometry:'rightArrow',position:{left:x,top:y,width:w,height:15},fill:C.red,line:{fill:'none',width:0}});}
function node(s,label,x,w){
  s.shapes.add({name:label+'-node',geometry:'rect',position:{left:x,top:117,width:w,height:58},fill:C.white,line:{fill:C.red,width:1.3}});
  text(s,label,label,x+5,124,w-10,44,24,C.red,false,FONT,'center');
}
function bars(s,name,x,categories,values){
  const c=s.charts.add('bar',{
    name,position:{left:x,top:180,width:390,height:230},
    categories,
    series:[{name:'相对 L₂ 误差',values,valuesFormatCode:'0.00%',fill:C.red,points:[{idx:0,fill:C.pale},{idx:1,fill:C.red}]}],
    hasLegend:false,barOptions:{direction:'column',grouping:'clustered',gapWidth:130},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00%',numberFormatSourceLinked:false,textStyle:{fontSize:23,fill:C.red,bold:true}},
    xAxis:{visible:true,textStyle:{fontSize:20,fill:C.gray},line:{fill:C.line,width:1},majorGridlines:null},
    yAxis:{visible:true,min:0,max:0.15,majorUnit:0.05,numberFormatCode:'0%',textStyle:{fontSize:16,fill:C.sub},line:{fill:'none',width:0},majorGridlines:{fill:C.line,width:0.7}},
    chartFill:C.white,chartLine:{fill:'none',width:0},plotAreaFill:C.white,plotAreaLine:{fill:'none',width:0}
  });
  for(const a of [c.xAxis.textStyle,c.yAxis.textStyle,c.dataLabels.textStyle,c.legend.textStyle]) a.typeface=FONT;
  return c;
}

const notes=[
`建议时长：50 秒。
这次我选取了一个具有解析解的一维热传导问题，完成标准 PINN 和混合 PINN 的基础复现。空间范围是负一到一，时间范围是零到一，热扩散系数固定为一。初始温度是正弦分布，两端温度为零。随着时间增加，温度分布的幅值按指数规律衰减。
这里的一维指只有一个空间坐标 x。方程里的二阶空间导数，是对同一个 x 连续求导两次，并不是二维空间。
解析解的作用是独立检验预测误差。训练使用方程、初值和边界约束，没有在内部点用解析解作为监督标签。下面重点介绍两种求导实现和对照结果。
来源：本项目 README.md、train_compare_demo2.py、outputs/results.json。背景与校徽来自用户提供的 PPTX，仅保留视觉素材。`,
`建议时长：65 秒。
两种方法使用同一个网络。每个输入由时间 t 和空间 x 组成，经过六个隐藏层，每层五十个神经元，使用 Tanh 激活，输出预测温度。可训练参数共 12951 个。
标准 PINN 通过自动微分得到时间一阶导数和空间二阶导数。混合 PINN 保留时间自动微分，用空间离散矩阵计算空间二阶导数。在均匀网格内部，三点公式是 (u[i-1]-2u[i]+u[i+1])/h²。如果 U 的每行代表一个时间的温度分布，则 Uxx=U D2 的转置。
两者都把残差 r=u_t-alpha*u_xx 的均方值，加上初值和边界损失进行训练。初值、边界权重都是 0.5，边界损失为左右两端 MSE 之和。Adam 初始学习率为 0.001，每一千步乘 0.5，总训练五千步。这些设置保持一致，才能比较求导方式。
来源：train_compare_demo2.py、train_hybrid_demo.py、outputs/results.json。`,
`建议时长：65 秒。
复现过程中，首先修正了原仓库的解析解。原来的初值是 sin(pi*x)，但参照解使用了另一组系数，误差不能正确反映求解结果。现在解析解、训练残差和评估使用同一个 alpha。
第二个关键点是混合方法的时间采样。空间差分衡量的是同一时刻沿空间的变化，所以一组空间点必须共享时间。原仓库逐点随机时间的做法，会把时间变化混入空间导数。
第三，网格和空间矩阵必须一同传入训练函数。五十个空间点对应四十八个内部导数，矩阵尺寸为四十八乘五十。端点通过边界损失约束，不套用不成立的端点求导模板。用常数、一阶多项式和 x 的平方做小检查，内部导数应分别为零、零和二。
来源：参考仓库 https://github.com/yydjya/PINN-solves-the-one-dimensional-heat-conduction-equation ，审查版本 e2e3a79b6eaaef08a22c9cf54bad3f161685155c；本项目混合实现及检查记录。`,
`建议时长：75 秒。
左图只比较求导方式。两种方法都使用四组时间，每组五十个空间点，每一步有一百九十二个内部残差。标准 PINN 的整体相对 L2 误差为 11.24%，混合为 11.03%，本次结果非常接近。对应耗时分别约为 42.28 秒和 26.10 秒，但混合耗时是从当时终端截图转录的，原截图没有纳入仓库，也没有重复计时，因此只能报告观察到的结果。
右图只比较标准方法的时间采样。保持空间坐标、内部残差数、初边值点、网络初始化和优化设置一致，给各 PDE 输入点独立抽取时间后，误差从 11.24% 降到 6.20%。这个结果提示，时间覆盖是值得继续研究的因素。不能把逐点独立时间直接用于混合方法的空间差分。
所有误差都在固定的一百零一个时间乘二百零一个空间位置上计算，指标为 ||预测-解析||2 / ||解析||2。这里只有 seed42 的一次结果，还不能判断统计意义上的普遍优劣。
来源：outputs/results.json；相关 metrics JSON。标准分组相对 L2=0.11237432062625885，混合分组=0.11031993478536606，标准独立时间=0.06203807145357132。`,
`建议时长：45 秒。
当前已经完成两种方法的基础实现、独立网格评估，以及一个控制变量的时间采样对照。标准方法采用逐点独立时间后，整体相对 L2 误差为 6.20%，还没有达到原先设定的百分之五目标。
这一组的初值最大误差为 0.0281，边界最大误差为 0.0496，均已满足 0.05 的目标。后续先补充 seed43 和重复计时，确认结论是否稳定。然后在保持空间组内同一时间的前提下增加时间覆盖，再评估 Adam 后接 L-BFGS 是否能继续降低误差。L-BFGS 目前属于待开展工作。
我目前的主要收获是：训练损失小并不等于全场误差小。复现需要先检查数学定义、采样和参照解的一致性，再讨论网络和优化器。
来源：outputs/results.json；独立时间结果的初值最大误差 0.0280982852，左边界 0.0495540015，右边界 0.0435759909。`
];

export async function buildPage(p,n){
  await fs.readFile(path.join(root,'spec_lock.md'),'utf8');
  let s;
  if(n===1){
    p.slides.keep([]);
    s=p.slides.add({width:W,height:H,background:{fill:C.white}});
    await image(s,'原模板建筑背景','image1.jpg',0,0,W,H);
    await image(s,'原模板 ECNU 校徽','image2.png',245.67,406.92,140.5,45.36);
    text(s,'cover-title','一维热传导 PINN 复现',240,145,640,58,38,C.red,true);
    text(s,'cover-subtitle','标准方法与混合方法的对照实验',240,208,620,40,25,C.red);
    text(s,'equation-domain','uₜ = αuₓₓ，α = 1     x ∈ [−1, 1]，t ∈ [0, 1]',240,270,650,32,22,C.gray,false,'Cambria Math');
    text(s,'initial-boundary','u(x, 0) = sin(πx)     u(−1, t) = u(1, t) = 0',240,307,650,32,22,C.gray,false,'Cambria Math');
    text(s,'exact-label','解析解',240,346,80,32,21,C.red);
    text(s,'exact','u(x, t) = exp(−απ²t) sin(πx)',323,346,550,32,23,C.gray,false,'Cambria Math');
    text(s,'meeting','组会汇报',555,414,220,36,22,C.red);
  }
  if(n===2){
    s=base(p,n,'标准 PINN 与混合 PINN');
    node(s,'输入 [t, x]',80,160); node(s,'6 × 50，Tanh',320,250); node(s,'温度 uθ(t, x)',650,230);
    arrow(s,257,139,47); arrow(s,587,139,47);
    text(s,'standard-title','标准 PINN',80,214,370,42,26,C.red,true);
    text(s,'standard-detail','uₜ、uₓₓ 均使用自动求导',80,265,400,42,23);
    text(s,'hybrid-title','混合 PINN',520,214,360,42,26,C.red,true);
    text(s,'hybrid-time','uₜ：自动求导',520,260,360,36,23);
    text(s,'hybrid-space','uₓₓ：同一时刻的空间三点差分',520,299,370,36,22);
    text(s,'residual','方程残差   r = uₜ − αuₓₓ',80,355,800,40,24,C.red,false,FONT,'center');
    text(s,'loss','总损失 = 方程损失 + 0.5 初值损失 + 0.5 边界损失',80,404,800,37,23,C.gray,false,FONT,'center');
    text(s,'optimizer','Adam：学习率 0.001，5000 步，每 1000 步学习率减半',80,461,800,32,20,C.sub,false,FONT,'center');
  }
  if(n===3){
    s=base(p,n,'原仓库的关键修正');
    const rows=[
      ['01','解析解与物理参数保持一致','解析解符合正弦初值，训练和评估使用同一个 α。'],
      ['02','空间差分只比较同一时刻的温度','一组 50 个空间点共享一个 t，避免混入时间变化。'],
      ['03','网格、矩阵与内部点保持一致','50 个空间点对应 48 × 50 的 D₂，端点由边界约束。']
    ];
    for(let i=0;i<rows.length;i++){
      const y=127+i*109;
      text(s,'correction-number-'+i,rows[i][0],80,y,52,50,33,C.red);
      text(s,'correction-title-'+i,rows[i][1],150,y,740,39,25,C.red,true);
      text(s,'correction-detail-'+i,rows[i][2],150,y+43,740,38,22);
    }
    text(s,'polynomial-check','小检查：1、x、x² 的内部离散二阶导分别为 0、0、2',80,468,800,32,20,C.sub);
  }
  if(n===4){
    s=base(p,n,'方法对照与时间采样实验');
    text(s,'fixed-settings','同一网络与优化设置，CPU，seed 42，5000 步',80,96,800,31,19,C.sub);
    text(s,'method-comparison-title','同一采样，比较求导方式',80,143,395,34,23,C.red,true);
    text(s,'sampling-comparison-title','同一标准方法，比较时间采样',505,143,395,34,23,C.red,true);
    bars(s,'求导方式相对误差',65,['标准 PINN','混合 PINN'],[0.11237432,0.11031993]);
    bars(s,'时间采样相对误差',495,['分组时间','逐点独立时间'],[0.11237432,0.06203807]);
    text(s,'timing','同组采样耗时：标准 42.28 s，混合 26.10 s*',80,428,800,33,21,C.gray);
    text(s,'evaluation-caveat','指标为固定 101 × 201 网格上的相对 L₂。单种子结果不代表普遍优劣。',80,474,800,22,14,C.sub);
    text(s,'timing-source','* 混合耗时由终端截图转录；尚未重复计时。',80,498,730,22,13,C.sub);
  }
  if(n===5){
    s=base(p,n,'基础复现完成，精度仍需改进');
    text(s,'completed-scope','已完成两种求解方法、独立网格评估及时间采样对照',80,101,800,34,21,C.sub);
    text(s,'current-relative-error','6.20%',80,155,250,80,60,C.red,true);
    text(s,'current-error-label','标准方法的整体相对 L₂ 误差',350,164,530,34,25,C.red);
    text(s,'target','逐点独立时间采样，目标 ≤ 5%',350,207,530,33,22,C.gray);
    text(s,'constraints-result','初值最大误差 0.0281；边界最大误差 0.0496',80,280,800,35,23,C.gray);
    text(s,'constraints-target','初边值均满足 0.05 目标，整体误差仍未达标',80,319,800,32,21,C.sub);
    text(s,'next-1','① 补充 seed 43 与重复计时，检查结果稳定性',80,382,800,34,23,C.gray);
    text(s,'next-2','② 增加时间覆盖，保持混合方法组内时间一致',80,425,800,34,23,C.gray);
    text(s,'next-3','③ 评估 Adam 后接 L-BFGS 的精度改进',80,468,800,34,23,C.gray);
  }
  s.speakerNotes.text=notes[n-1];
  await fs.writeFile(path.join(root,'notes',String(n).padStart(2,'0')+'.md'),notes[n-1],'utf8');
  return {page:n,id:s.id,frame:s.frame};
}

export { root, W, H };
