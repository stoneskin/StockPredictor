
```
// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/

// © stoneskin

  

//@version=5

strategy("My Predict strategy for MACD+RSI", "myPredictMACDRSI", overlay=true,initial_capital = 10000,default_qty_type = strategy.percent_of_equity,default_qty_value = 50,process_orders_on_close = true )

  

useDateFilterEnd = input.bool(false, title="End Backtest at The End Date", group="Backtest Time Period")

trainingStartDate = input.time(timestamp("2 Jan 2020"),

     title="Training Start", group="Backtest Time Period",

     tooltip="This start date is in the time zone of the exchange " )

backtestStartDate = input.time(timestamp("2 Jan 2021"),

     title="Trade Start Date", group="Backtest Time Period",

     tooltip="This start date is in the time zone of the exchange ")

backtestEndDate = input.time(timestamp("5 Jan 2025"),

     title="End Date", group="Backtest Time Period",

     tooltip="This End date is in the time zone of the exchange " )

inTradeWindow =  time >= backtestStartDate  

inTrainingWindow= time>= trainingStartDate

if (useDateFilterEnd)

    inTradeWindow:=inTradeWindow and time<=backtestEndDate

    inTrainingWindow:=inTrainingWindow and time <=trainingStartDate

  

tradingDirection=input.int(defval = 1,minval = -1,maxval = 1,title = "long/short/both",tooltip = "-1=short,-1=long,0=both")

tradingLong=tradingDirection>-1

tradingShort=tradingDirection<1

  

price=input(close, title='Price')

  
  

fast_length = input(title="Fast Length", defval=12)

slow_length = input(title="Slow Length", defval=26)

signal_length = input.int(title="Signal Smoothing",  minval = 1, maxval = 50, defval = 9)

sma_source = input.string(title="Oscillator MA Type",  defval="EMA", options=["SMA", "EMA"])

sma_signal = input.string(title="Signal Line MA Type", defval="EMA", options=["SMA", "EMA"])

  
  

fast_ma = sma_source == "SMA" ? ta.sma(price, fast_length) : ta.ema(price, fast_length)

slow_ma = sma_source == "SMA" ? ta.sma(price, slow_length) : ta.ema(price, slow_length)

macd = fast_ma - slow_ma

signal = sma_signal == "SMA" ? ta.sma(macd, signal_length) : ta.ema(macd, signal_length)

hist = macd - signal

  
  

//Plot colors

col_macd12 = #2962FF

col_signal12 = #FF6D00

col_grow_above12 = #26A69A

col_fall_above12 = #B2DFDB

col_grow_below12 = #FFCDD2

col_fall_below12 = #FF5252

  
  
  
  

bullTrand="-" //macd status

if(macd>0 and signal>0)

    bullTrand:="1"

else if(macd<0 and signal<0)

    bullTrand:="0"

else if(signal>0)

    bullTrand:='1'

else

    bullTrand:='0'

  

macdStatus='0'

if(macd>signal)

    macdStatus:='1'

  
  
  
  

// todo: add hist signal

  
  
  
  

//add RSI

smoothK = input.int(3, "K", minval=1)

smoothD = input.int(3, "D", minval=1)

lengthRSI = input.int(14, "RSI Length", minval=1)

lengthStoch = input.int(14, "Stochastic Length", minval=1)

  

rsi1 = ta.rsi(price, lengthRSI)

k = ta.sma(ta.stoch(rsi1, rsi1, rsi1, lengthStoch), smoothK)

d = ta.sma(k, smoothD)

//plot(k, "K", color=#2962FF)

//plot(d, "D", color=#FF6D00)

//h0 = hline(80, "Upper Band", color=#787B86)

//hline(50, "Middle Band", color=color.new(#787B86, 50))

//h1 = hline(20, "Lower Band", color=#787B86)

//fill(h0, h1, color=color.rgb(33, 150, 243, 90), title="Background")

  

rsiStatus="0"

if(k>d)

    rsiStatus:="1"

  

rsiOverStatus="-"

if(k>80)

    rsiOverStatus:="1"

if(k<20)

    rsiOverStatus:="0"

  
  
  

//build keys

var predKeys=""

predKeys:="P_"+str.tostring(bullTrand)+macdStatus+rsiStatus+rsiOverStatus

  

predKeyChanged=not str.contains(predKeys[1],predKeys)

predKeys:=predKeys+str.tostring(predKeyChanged)

  
  

/////////////////////////////////////////////////////////////////////////////

//start pred code

//define a dictionary

  

var  hisScoresKey  = array.new_string(0)

var  hisScoreWinCt = array.new_int(0)

var  hisScoreLossCt = array.new_int(0)

var  hisScoreMaxWin = array.new_float(0)

var  hisScoreMaxLoss = array.new_float(0)

var  hisScoreAvgWin = array.new_float(0)

var  hisScoreAvgLoss = array.new_float(0)

  
  
  

hisScores_getValue (score) =>

    string key=str.tostring(score)

    int ind=array.indexof(hisScoresKey,key)

    int winCt=na

    int lossCt=na

    float maxWin=na

    float maxLoss=na

    float avgWin=na

    float avgLoss=na

  

    if(ind>-1)    

        winCt:=array.get(hisScoreWinCt, ind)

        lossCt:=array.get(hisScoreLossCt, ind)

        maxWin:=array.get(hisScoreMaxWin, ind)

        maxLoss:=array.get(hisScoreMaxLoss, ind)

        avgWin:=array.get(hisScoreAvgWin, ind)

        avgLoss:=array.get(hisScoreAvgLoss, ind)

    [ind,winCt,lossCt,maxWin,maxLoss,avgWin,avgLoss]

  

hisScores_setValue (score, p0,p5) => //pass in price current and price 5 day

    string key=str.tostring(score)

    int winCt=p0>p5?1:0

    int lossCt=p0<p5?1:0

    float win=math.round(100*(p0-p5)/p5,2)

    float loss=math.round(100*(p0-p5)/p5,2)  

  

    int ind=array.indexof(hisScoresKey,key)

  

    if(ind==-1)

        array.push(hisScoresKey, key)

        array.push(hisScoreWinCt , winCt)

        array.push(hisScoreLossCt , lossCt)

        array.push(hisScoreMaxWin ,win)

        array.push(hisScoreMaxLoss , loss)

        array.push(hisScoreAvgWin ,win)

        array.push(hisScoreAvgLoss , loss)

    else

        [pInd,pWinCt,pLossCt,pMaxWin,pMaxLoss,pAvgWin,pAvgLoss]=hisScores_getValue (key)

        wCt=winCt+pWinCt

        lCt=lossCt+pLossCt

        avgWin=wCt>0?(pAvgWin*pWinCt+win)/wCt:0

        avgLoss=lCt>0?(pAvgLoss*pLossCt+loss)/lCt:0

        array.set(hisScoreWinCt , ind, wCt)

        array.set(hisScoreLossCt , ind, lCt)

        if(win>pMaxWin)

            array.set(hisScoreMaxWin , ind, win)

        if(loss<pMaxLoss)

            array.set(hisScoreMaxLoss , ind, loss)

        array.set(hisScoreAvgWin , ind, avgWin)

        array.set(hisScoreAvgLoss , ind, avgLoss)

  

predLength = input.int(defval=10, minval=2, maxval=100, title='pre check length',group="Predict")

predUseInTradeWindow=input.bool(defval = true,title = "use InTrade start time",group="Predict")

predShowDetail=input.bool(defval = false,title = "show pred details",group="Predict")

//predShowInChart=input.bool(defval = false,title = "show pred in chart",group="Predict")

  

ema_pred = ta.ema(price, predLength)

if(bar_index>predLength+1 and inTrainingWindow)

    hisScores_setValue(predKeys[predLength],math.round(ema_pred,2),price[predLength])

  

var float upRate=na

upRate:=0

  

bool longCondition=na

bool shortCondition=na

actionMsg=""

boxMsg = "=====================================\n"

boxMsg:=boxMsg+predKeys+"  -->  "

if(bar_index>1000 and predUseInTradeWindow?inTradeWindow:true)

    [ind,winCt,lossCt,maxWin,maxLoss,avgWin,avgLoss] = hisScores_getValue (predKeys)

    if(ind>-1)

        upRate:=math.round(100*(winCt-lossCt)/(winCt+lossCt),2)

        if(str.contains(predKeys,"true")) //not count none cross over bar

            if(upRate>10 and tradingLong)

                longCondition:=true

            if(upRate<0 and tradingLong)

                longCondition:=false        

            if(upRate<-10 and tradingShort)

                shortCondition:=true

            if(upRate>0 and tradingShort)

                shortCondition:=false

  
  
  

            //logic #2          

            // if(macdStatus=="1" and rsiStatus=="1" and tradingLong)

            //     if rsiOverStatus!="1"

            //         longCondition:=true

            // else if tradingLong

            //     longCondition:=false

  
  

            // if(macdStatus=="0" and rsiStatus=="0"  and tradingShort)

            //     if rsiOverStatus!="0"

            //         shortCondition:=true

            // else if tradingShort

            //     shortCondition:=false

            //logic #3

            // if(macdStatus=="1" and rsiStatus=="1")

            //     longCondition:=true

            //     shortCondition:=false

  

            // if(macdStatus=="0" and rsiStatus=="0")

            //     shortCondition:=true            

            //     longCondition:=false

  

        boxMsg:=boxMsg+str.tostring(upRate)+ "%  long="+str.tostring(longCondition) +" short="+str.tostring(shortCondition)+"\n"

        //boxMsg:=boxMsg+"barIndex="+str.tostring(bar_index)+",isNew="+str.tostring(barstate.isnew)+",isLast="+str.tostring(barstate.islast)+",isLfh="+str.tostring(barstate.islastconfirmedhistory)+"\n"

        upMsg="upRate="+ str.tostring(math.round(100*winCt/(winCt+lossCt),2))+"%("+str.tostring(winCt)+"), maxWin="+str.tostring(math.round(maxWin,2))+"% ,avgWin="+str.tostring(math.round(avgWin,2))+"%\n"

        downMsg="downRate="+ str.tostring(math.round(100*lossCt/(winCt+lossCt),2))+"%("+str.tostring(lossCt)+"), maxLoss="+str.tostring(math.round(maxLoss,2))+"% ,avgLoss="+str.tostring(math.round(avgLoss,2))+"%\n"

        boxMsg:=boxMsg+"hisScoresKey size="+str.tostring(array.size(hisScoresKey))+", len="+str.tostring(predLength)+",\n"+upMsg+downMsg

        actionMsg:=predKeys+" "+str.tostring(upRate)+"%("+str.tostring(winCt)+":"+str.tostring(lossCt)+")"

  
  
  

if(barstate.islastconfirmedhistory and inTrainingWindow and predShowDetail) //add time consume on last bar only

    boxMsg:=boxMsg+"=====================================\n"

    lab = ""

    for i = 0 to array.size(hisScoresKey) - 1

        dwinCt=array.get(hisScoreWinCt,i)

        dlossCt=array.get(hisScoreLossCt,i)

        dmaxWin=array.get(hisScoreMaxWin, i)

        dmaxLoss=array.get(hisScoreMaxLoss, i)

        davgWin=array.get(hisScoreAvgWin, i)

        davgLoss=array.get(hisScoreAvgLoss, i)

        dupRate =math.round(100*(dwinCt-dlossCt)/(dwinCt+dlossCt),2)

  

        if(math.abs(dupRate)>0 and dwinCt+dlossCt>0)

            lab := lab+"i="+str.tostring(i)+" "

            lab := lab + array.get(hisScoresKey, i) + "  upRate="+str.tostring(dupRate)+"%"

            lab := lab+"  winCt="+str.tostring(dwinCt) +" lossCt="+str.tostring(dlossCt)

            //lab := lab+"  maxW="+str.tostring(math.round(dmaxWin,2)) +"% maxL="+str.tostring(math.round(dmaxLoss,2)) +"%"

            //lab := lab+"  avgW="+str.tostring(math.round(davgWin,2)) +"% avgL="+str.tostring(math.round(davgLoss,2)) +"% \n"

            lab:=lab+"\n"

    boxMsg:=boxMsg+lab

  

myBox=box.new(left=bar_index+10,top=(high+20)*1.1,right=bar_index+75,bottom=(low-20)*0.9,bgcolor =color.new(color.purple, 70), border_width=0,text = boxMsg,text_color = color.white,text_size=size.normal,text_wrap = text.wrap_none,text_halign = text.align_left)

box.delete(myBox[1])

//end of pred code

////////////////////////////////////////////////////////////////////////////////

  
  

plotchar(longCondition, "Long", "▲", location.belowbar, color =  upRate>30?color.orange: color.blue, size = size.normal)

plotchar(shortCondition, "Short", "▼", location.abovebar, color = upRate<-10? color.fuchsia:color.gray, size = size.normal)

  

labelColor=upRate<-30?color.fuchsia:upRate<-10?color.aqua:upRate<10?color.silver:upRate<30?color.blue:color.orange

if( str.endswith(predKeys,"true"))

    if(upRate>0)

        label.new(bar_index,math.max(low,low[1])*0.95,actionMsg,color = labelColor,style = label.style_label_up)

    else

        label.new(bar_index,math.min(high,high[1])*1.05,actionMsg, color = labelColor,style = label.style_label_down)

  
  

// longCondition = ta.crossover(ta.sma(close, 14), ta.sma(close, 28))

  

if (longCondition==true and tradingLong)

    strategy.entry("Long", strategy.long,stop=price*0.97,comment = actionMsg)

if(longCondition==false and tradingLong)

    strategy.close("Long",comment = actionMsg)

  

// shortCondition = ta.crossunder(ta.sma(close, 14), ta.sma(close, 28))

if (shortCondition==true and tradingShort)

    strategy.entry("Short", strategy.short,stop = price*1.03,comment = actionMsg)

if(shortCondition==false and tradingShort)

    strategy.close("Short",comment = actionMsg)
```


## Pine Script Review Summary

### **What It Is**

This is a **trading strategy called "My Predict strategy for MACD+RSI"** that backtests on historical data to find profitable signal combinations, then uses those results to make trading decisions.

---

### **Key Components**

1. **Technical Indicators:**
    
    - **MACD** (Fast: 12, Slow: 26) - Identifies momentum direction
    - **RSI + Stochastic** (RSI Length: 14, Stoch Length: 14) - Measures overbought/oversold conditions
2. **Signal Generation:**
    
    - Creates a "prediction key" combining:
        - MACD trending status (bullish/bearish)
        - MACD crossover signal
        - RSI/Stochastic K>D condition
        - Overbought/oversold status
        - Whether this is a new signal combination
3. **Historical Scoring System:**
    
    - Tracks every signal combination that appears during the training period
    - Records win rate, loss rate, max profit, max loss, and average profit/loss for each combination
    - Rates signals with a "win-loss ratio" percentage
4. **Trade Execution Logic:**
    
    - **Long**: Enters when current signal has >10% win rate; exits when win rate drops to 0% or negative
    - **Short**: Enters when current signal has <-10% win rate; exits when win rate goes positive
    - Uses stop losses at 3% (3% below entry for longs, 3% above for shorts)

---

### **How to Use It**

**1. Configure Backtest Period:**

- Set "Training Start" date (e.g., 2 Jan 2020) - learns signal patterns from this period
- Set "Trade Start Date" (e.g., 2 Jan 2021) - starts trading from this date
- Optional: Set "End Date" to limit backtest window

**2. Set Trading Mode:**

- Input "long/short/both": -1 (short only), 0 (both), 1 (long only)

**3. Fine-tune Indicators (Optional):**

- Adjust MACD fast/slow lengths, signal smoothing
- Adjust RSI and Stochastic lengths
- Change "pre check length" (lookback period for EMA calculation, default 10 bars)

**4. View Results:**

- **Triangles on chart**: ▲ = long signal, ▼ = short signal
- **Purple box**: Shows current signal key and win-loss percentage
- Enable "show pred details" to see all signal combinations and their performance

**5. How It Works:**

- First 1000 bars: Training phase (accumulating signal statistics)
- After bar 1000: Trading phase (using the trained signal statistics)
- The script only trades signals that showed a win rate exceeding ±10%

---

### **Potential Issues**

- Strategy relies on historical patterns repeating (no guarantee in live trading)
- Has commented-out alternative trading logic (#2 and #3) that could be experimented with
- The prediction threshold (10% win rate) is hardcoded

This is an **algorithmic backtesting strategy** useful for testing if certain technical indicator combinations historically predict profitable moves.