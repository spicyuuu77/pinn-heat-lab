import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const ROOT=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const font='DengXian', red='#A51E36', gray='#555759', secondary='#6C6E70', grid='#D8D8D8';

function text(s,name,value,x,y,w,h,size=23,color=gray,bold=false){
  const sh=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  sh.text=value;
  sh.text.style={typeface:font,fontSize:size,color,bold,insets:0,wrap:'none',verticalAlignment:'middle'};
  return sh;
}
function chartFont(c){
  for(const a of [c.xAxis.textStyle,c.yAxis.textStyle,c.legend.textStyle,c.dataLabels.textStyle])a.typeface=font;
  for(const a of [c.xAxis,c.yAxis])if(a.title?.textStyle)a.title.textStyle.typeface=font;
}

export async function addResults(p){
  await fs.readFile(path.join(ROOT,'spec_lock.md'),'utf8');
  const data=JSON.parse(await fs.readFile(path.join(ROOT,'build/result_chart_data.json'),'utf8'));
  if(p.slides.items.length!==5)throw new Error('Expected the five-slide source deck');
  const previous=p.slides.items.map(sl=>'sl/'+sl.id);
  const conclusion=p.slides.items[4];
  const s=p.slides.add({width:960,height:5184775/9525,background:{fill:'#FFFFFF'}});
  text(s,'page-title','温度预测与训练过程',80,44,810,48,32,red);
  text(s,'result-run','标准 PINN，逐点独立时间，seed 42，5000 步',80,98,810,31,19,secondary);
  text(s,'temperature-title','三个时刻的温度分布',80,139,400,34,23,red,true);
  text(s,'loss-title','训练总损失',530,139,350,34,23,red,true);
  const series=[];
  const colors=[red,'#355E7C',gray];
  data.slices.forEach((slice,i)=>{
    series.push({name:`解析 t=${slice.t}`,xValues:data.x,values:slice.exact,xValuesFormatCode:'0.00',valuesFormatCode:'0.00000000',line:{style:'solid',fill:colors[i],width:1.8},marker:{symbol:'none'}});
    series.push({name:`预测 t=${slice.t}`,xValues:data.x,values:slice.predicted,xValuesFormatCode:'0.00',valuesFormatCode:'0.00000000',line:{style:'dashed',fill:colors[i],width:1.6},marker:{symbol:'none'}});
  });
  const temp=s.charts.add('scatter',{
    position:{left:55,top:181,width:425,height:283},series,
    scatterOptions:{style:'line'},hasLegend:true,
    legend:{position:'bottom',overlay:false,textStyle:{fontSize:14,fill:gray}},
    xAxis:{visible:true,position:'bottom',tickLabelPosition:'low',crossesAt:-1,min:-1,max:1,majorUnit:0.5,numberFormatCode:'0.0',title:{text:'x',textStyle:{fontSize:18,fill:gray}},textStyle:{fontSize:15,fill:secondary},line:{fill:grid,width:0.8},majorGridlines:null},
    yAxis:{visible:true,position:'left',tickLabelPosition:'low',crossesAt:-1.2,min:-1.2,max:1.2,majorUnit:0.6,numberFormatCode:'0.0',title:{text:'u',textStyle:{fontSize:18,fill:gray}},textStyle:{fontSize:15,fill:secondary},line:{fill:'none',width:0},majorGridlines:{fill:grid,width:0.6}},
    chartFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}
  });
  chartFont(temp);
  const loss=s.charts.add('scatter',{
    position:{left:505,top:181,width:390,height:283},
    series:[{name:'log10(总损失)',xValues:data.steps,values:data.log10_total,xValuesFormatCode:'0',valuesFormatCode:'0.00000000',line:{style:'solid',fill:red,width:0.8},marker:{symbol:'none'}}],
    scatterOptions:{style:'line'},hasLegend:false,
    xAxis:{visible:true,position:'bottom',tickLabelPosition:'low',crossesAt:0,min:0,max:5000,majorUnit:2500,numberFormatCode:'0',title:{text:'训练步数',textStyle:{fontSize:17,fill:gray}},textStyle:{fontSize:15,fill:secondary},line:{fill:grid,width:0.8},majorGridlines:null},
    yAxis:{visible:true,position:'left',tickLabelPosition:'low',crossesAt:-4,min:-4,max:0,majorUnit:1,numberFormatCode:'0',title:{text:'log₁₀(总损失)',textStyle:{fontSize:17,fill:gray}},textStyle:{fontSize:15,fill:secondary},line:{fill:'none',width:0},majorGridlines:{fill:grid,width:0.6}},
    chartFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}
  });
  chartFont(loss);
  text(s,'legend-explanation','左图：实线为解析解，虚线为预测。右图：纵轴每下降 1，损失缩小 10 倍。',80,474,820,24,16,secondary);
  text(s,'result-error','固定时空网格上的相对 L₂ 误差仍为 6.20%，不能只凭曲线接近判断全域精度。',80,501,790,23,14,secondary);
  text(s,'page-number','05',858,510,30,18,13,secondary);
  const pageNum=conclusion.shapes.items.find(a=>a.name==='page-number');
  if(pageNum)pageNum.text='06';
  p.slides.reorder([...previous.slice(0,4),'sl/'+s.id,previous[4]]);
  s.speakerNotes.text=`建议时长：50 秒。
这一页展示实际计算结果，全部取自标准 PINN 使用逐点独立时间采样的同一次实验，seed42、5000 步。
左图选取 t=0、0.1 和 0.3 三个时刻。相同颜色表示相同时间，实线为解析解，虚线为网络预测。可以看到温度分布的正弦形状以及随时间衰减的趋势。曲线接近说明这几个切片的拟合情况，但不能代替全域误差检查。
右图来自保存的五千步原始总损失，每一步都保留，没有平滑或抽样。为了看清数量级变化，绘制 log10(total loss)，所以纵轴下降一个单位相当于损失缩小十倍。总损失包含 PDE、0.5 倍初值损失和 0.5 倍边界损失。
这次从保存权重重新评估，101×201 网格上的相对 L2 为 ${data.metrics.relative_L2.toFixed(8)}，与原记录一致，显示为 6.20%。
来源：${data.model}；${data.history}；outputs/results.json。曲线数据仅为图表存储四舍五入至小数点后八位，原实验文件保持不变。`;
  await fs.writeFile(path.join(ROOT,'notes','05_results.md'),s.speakerNotes.text,'utf8');
  return {slideCount:p.slides.items.length,newSlide:s.id};
}
