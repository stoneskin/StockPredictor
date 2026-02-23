```
// modified by stoneskin 2022-11-04

//@version=5

  

strategy('Vegas Channel + Hull STRG', shorttitle='VegasChannel+Hull', overlay=true,initial_capital = 10000,default_qty_type = strategy.cash,default_qty_value = 5000,process_orders_on_close = true)

lenFilter=input(12,title='过滤线')

emaFilter = ta.ema(close, lenFilter)

plot(emaFilter, title='过滤线', color=color.yellow, linewidth=2)

  

multiFast = input.int(1,title = "multiplier fast",minval=1,maxval=10,tooltip = "try 1 for 1d; 2 for 2~4h; 3,4,8 for 1h; 1,2,8 for 30m; 2,3,5,6 for 15m; 1,4,8 for 5min;")

multiSlow = input.int(8,title = "multiplier slow",minval=1,maxval=10,tooltip = "try 1 for 1d; 2 for 2~4h; 3,4,8 for 1h; 1,2,8 for 30m; 2,3,5,6 for 15m; 1,4,8 for 5min;")

  

//MA1

MA1Period = input.int(title='MA1 Period', defval=144, minval=1)*multiFast

//MA1Type = input.string(title='MA1 Type', defval='EMA', options=['SMA', 'EMA', 'WMA', 'RMA'])

MA1Source = input(title='MA1 Source', defval=close)

MA1Visible = input(title='MA1 Visible', defval=true)

  

//MA2

MA2Period = input.int(title='MA2 Period', defval=169, minval=1)*multiFast

//MA2Type = input.string(title='MA2 Type', defval='EMA', options=['SMA', 'EMA', 'WMA', 'RMA'])

MA2Source = input(title='MA2 Source', defval=close)

MA2Visible = input(title='MA2 Visible', defval=true)

  

//MA1-2

MA12Period = input.int(title='MA12 Period', defval=144*4, minval=1)*multiFast

//MA12Type = input.string(title='MA12 Type', defval='EMA', options=['SMA', 'EMA', 'WMA', 'RMA'])

MA12Source = input(title='MA12 Source', defval=close)

MA12Visible = input(title='MA12 Visible', defval=true)

  

//MA2-2

MA22Period = input.int(title='MA22 Period', defval=169*4, minval=1)*multiFast

//MA22Type = input.string(title='MA22 Type', defval='EMA', options=['SMA', 'EMA', 'WMA', 'RMA'])

MA22Source = input(title='MA22 Source', defval=close)

MA22Visible = input(title='MA22 Visible', defval=true)

  

//MA3

MA3Period = input.int(title='MA3 Period', defval=288, minval=1)*multiSlow

MA3Type = input.string(title='MA3 Type', defval='EMA', options=['SMA', 'EMA', 'WMA', 'RMA'])

MA3Source = input(title='MA3 Source', defval=close)

MA3Visible = input(title='MA3 Visible', defval=true)

  

//MA4

MA4Period = input.int(title='MA4 Period', defval=338, minval=1)*multiSlow

MA4Type = input.string(title='MA4 Type', defval='EMA', options=['SMA', 'EMA', 'WMA', 'RMA'])

MA4Source = input(title='MA4 Source', defval=close)

MA4Visible = input(title='MA4 Visible', defval=true)

  
  

//MA1 is series type

MA1 =  ta.ema(MA1Source, MA1Period)

  

MA2 = ta.ema(MA2Source, MA2Period)

  

MA12 =  ta.ema(MA12Source, MA12Period)

  

MA22 = ta.ema(MA22Source, MA22Period)

  
  
  

MA3 = if MA3Type == 'SMA'

    ta.sma(MA3Source, MA3Period)

else

    if MA3Type == 'EMA'

        ta.ema(MA3Source, MA3Period)

    else

        if MA3Type == 'WMA'

            ta.wma(MA3Source, MA3Period)

        else

            if MA3Type == 'RMA'

                ta.rma(MA3Source, MA3Period)

  

MA4 = if MA4Type == 'SMA'

    ta.sma(MA4Source, MA4Period)

else

    if MA4Type == 'EMA'

        ta.ema(MA4Source, MA4Period)

    else

        if MA4Type == 'WMA'

            ta.wma(MA4Source, MA4Period)

        else

            if MA4Type == 'RMA'

                ta.rma(MA4Source, MA4Period)

  

period=input.int(3,"period to get slop")

periodMult=input.int(3,"*Period to get avg slop")

thresholdSlop=input.float(3,minval = 0, title =  "Trend threshold")

sMa=ta.sma(MA1,period)

slop=1000*(sMa-sMa[period])/sMa

maSlop=ta.sma(slop, period*periodMult)

fColor=math.abs(maSlop)>thresholdSlop? maSlop>0? color.lime:color.red:color.fuchsia

//plot(sMa , color=color.aqua, linewidth=2)

//plot(maSlop , color=fColor, linewidth=5)

  

p1=plot(MA1Visible ? MA1 : na, "p1", color=fColor, linewidth=1)

p2=plot(MA2Visible ? MA2 : na, "p2",color=fColor, linewidth=1)

fill(p1, p2, title='Band Filler', color=fColor)

//hullColor =  color.new( MA1>MA2 ?#00ff00 : #ff0000 ,60)

fColor12 =  color.new( MA12>MA22 ?color.green :color.purple,60)

p12=plot(MA12Visible ? MA12 : na, "p12", color=fColor12, linewidth=1)

p22=plot(MA22Visible ? MA22 : na, "p22",color=color.new(color.silver, 0), linewidth=1)

fill(p12, p22, title='Band Filler', color=fColor12)

  

p3=plot(MA3Visible ? MA3 : na, color=color.new(color.white, 0), linewidth=1)

p4=plot(MA4Visible ? MA4 : na, color=color.new(color.silver, 0), linewidth=1)

fColor2 =  color.new( color.gray ,80)

fill(p3, p4, title='Band Filler', color=fColor2)

  

Channel_Allow_Long= MA1>MA12 and MA1>MA22 and close>MA1 and close>MA12 and fColor==color.lime

Channel_Allow_Short=MA1<MA12 and MA1<MA22 and  close<MA1 and close<MA12 and fColor==color.red

Channel_Close_Long= close<MA1 and close <MA2 and close<MA12 and close <MA22

Channel_Close_Short=close>MA1 and close>MA2 and close>MA12 and close>MA22

//***************************Hull suite (HMA) *****************************************//

//FUNCTIONS

//HMA

int useHull =input.int(1,title="useHull 0=NotUse,1=HML,2,EHMA,3,THMA ",group = "Hull suite")

// bool useHullAsFilter=input.bool(false,"User Hull As Open Filter",group = "Hull suite" )

// bool useHullOpen=input.bool(false,"User Hull As Open check",group = "Hull suite" )

// useHullasExit=input.bool(true,"Use Hull+SAR as Exit",group =  "BackTest")

HMA(_src, _length) =>

    ta.wma(2 * ta.wma(_src, _length / 2) - ta.wma(_src, _length), math.round(math.sqrt(_length)))

//EHMA    

EHMA(_src, _length) =>

    ta.ema(2 * ta.ema(_src, _length / 2) - ta.ema(_src, _length), math.round(math.sqrt(_length)))

//THMA    

THMA(_src, _length) =>

    ta.wma(ta.wma(_src, _length / 3) * 3 - ta.wma(_src, _length / 2) - ta.wma(_src, _length), _length)

//SWITCH

HullMode(modeSwitch, src, len) =>

    modeSwitch == 1 ? HMA(src, len) : modeSwitch == 2 ? EHMA(src, len) : modeSwitch == 3 ? THMA(src, len / 2) : na

  

hma_src=close

//hma_modeSwitch = input.string('Hma', title='Hull Variation', options=['Hma', 'Thma', 'Ehma'], group="Hull suite")

hma_length = input(55, title='HMA lentgh', tooltip = "Length(180-200 for floating S/R , 55 for swing entry)", group="Hull suite")

hma_lengthMult = input(1.0, title='HMA mutiple',tooltip = "Length multiplier (Used to view higher timeframes with straight band)", group="Hull suite")

  

//OUT

  

float HULL = HullMode(useHull,hma_src, int(hma_length * hma_lengthMult))//Mode(hma_modeSwitch, hma_src, int(hma_length * hma_lengthMult))

MHULL = HULL[0]

SHULL = HULL[2]

  

bool hull_signal=HULL > HULL[2]

  

hullColor =  color.new( hull_signal ?#00ff00 : #ff0000 ,60)

Fi1 = plot(MHULL, title='MHULL', color=hullColor, linewidth=1)

Fi2 = plot(SHULL, title='SHULL', color=hullColor, linewidth=1)

fill(Fi1, Fi2, title='Band Filler', color=hullColor)

  

Hull_Allow_Long=hull_signal and low>MHULL and low> SHULL

Hull_Allow_Short=hull_signal==false and high<MHULL and high<SHULL

//***************************End Hull suite (HMA)*****************************************//

  
  

///////////////////////////////backtest//////////////////////////////////

//date time range

backtestStartDate = input.time(timestamp("1 Oct 2021"),

     title="Trade Start Date", group="Backtest Date",

     tooltip="This start date is in the time zone of the exchange ")

backtestEndDate = input.time(timestamp("5 Jan 2025"),

     title="End Date", group="Backtest Date",

     tooltip="This End date is in the time zone of the exchange " )

bool useEndDate=input.bool(false,tooltip = "Use back test end date ", group="Backtest Date")

inTradeWindow =  (time >= backtestStartDate) and (useEndDate? (time<=backtestEndDate):true)

  

//Action

Action_Allow_Long=Channel_Allow_Long and Hull_Allow_Long

Action_Allow_Short=Channel_Allow_Short and Hull_Allow_Short

  

action_signal="action|long:"+  str.tostring(Action_Allow_Long)+"|short:"+str.tostring(Action_Allow_Short)

signalStatus=str.tostring(Channel_Allow_Long)+str.tostring(Hull_Allow_Long)+"|"+str.tostring(Channel_Allow_Short)+str.tostring(Hull_Allow_Short)

  

bool Buy=na

bool Sell=na

bool CloseLong=na

bool CloseShort=na

//var float stopLoss=na

if action_signal != action_signal[1] and inTradeWindow

    Buy:= Action_Allow_Long and strategy.position_size<=0

    Sell:=Action_Allow_Short and strategy.position_size>=0

  

if strategy.position_size>0

    CloseLong:=   Channel_Close_Long  or Hull_Allow_Short

if strategy.position_size<0  

    CloseShort:= Hull_Allow_Long or Channel_Close_Short

  

plotchar(Buy, "Long", "▲", location.belowbar, color = color.yellow, size = size.small)

plotchar(Sell, "short", "▼", location.abovebar, color = color.blue, size = size.small)

  

winloss=close>strategy.position_avg_price

  
  

plotchar(CloseLong, "Long", "X", location.belowbar, color = winloss==true? color.green:color.red, size = size.small)

plotchar(CloseShort, "short", "X", location.abovebar, color = winloss==true?color.blue: color.fuchsia, size = size.small)

  
  

if Buy and strategy.position_size<=0

    strategy.entry("Long", strategy.long,comment = "buy"+signalStatus)

    //stopLoss:=  math.max(ta.lowest(low,5), close*0.97 )

if Sell and strategy.position_size>=0

    strategy.entry("Short", strategy.short,comment ="sell"+signalStatus)

    //stopLoss:= math.min(ta.highest(high,5),close*1.03)

  

if strategy.position_size>0 and ( CloseLong)

    strategy.close("Long", comment = "stop:"+signalStatus)

if strategy.position_size<0 and ( CloseShort)

    strategy.close("Short", comment = "stop:"+signalStatus)
```

## Script Review: Vegas Channel + Hull STRG

**Overview:** This is a TradingView trading strategy that combines Vegas Channel technical analysis with Hull Moving Average (HMA) indicators to generate long/short trading signals.

### Main Components:

**1. Vegas Channel System**

- Uses 4 exponential moving averages (MA1, MA2, MA12, MA22) to create price channels
- MA1/MA2 form a fast channel (144 & 169 period by default)
- MA12/MA22 form a slower channel (576 & 676 period by default)
- Channel color changes based on momentum (lime for bullish, red for bearish, fuchsia for neutral)

**2. Trend Filtering**

- EMA filter line (12-period, adjustable) for basic trend direction
- Slope detection calculates momentum by measuring rate of change in the MA1 slope
- Threshold-based trend confirmation (3 points default)

**3. Hull Moving Average Suite**

- Implements 3 Hull variants: HMA, EHMA (exponential), THMA (triple)
- 55-period default with multiplier option for higher timeframe analysis
- Shows support/resistance bands

**4. Trading Signals**

- **Long Entry:** When MA1 > MA12, close > MA1, AND Hull confirms (uptrend)
- **Short Entry:** When MA1 < MA12, close < MA1, AND Hull confirms (downtrend)
- **Exit Long:** When close drops below MA1/MA2/MA12/MA22 OR Hull turns downtrend
- **Exit Short:** When close rises above price levels OR Hull turns uptrend

**5. Backtesting Parameters**

- Configurable date range (Oct 2021 - Jan 2025 by default)
- Initial capital: $10,000
- Risk per trade: $5,000 (adjustable)

### What It Does:

This is a **trend-following strategy** that uses dual confirmation (Vegas channels + Hull MA) to reduce false signals. It's designed to catch swing trades in both directions with built-in support/resistance markers for better entry/exit timing.