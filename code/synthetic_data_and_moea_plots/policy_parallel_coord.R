### parallel axis plot with densities, for DPS control policies
library(reshape2)
library(ggplot2)
library(grid)
library(gridExtra)
library(gtable)
library(shadowtext)
library(plyr)
library(plotly)

dir_data <- '../../data/policy_simulation/4obj/'
dir_figs <- '../../figures/'


# input data
df <- read.csv(file=paste(dir_data, 'mi_examples_simulation.csv', sep=''), header=TRUE, sep=",", stringsAsFactors = F)
#df <- df[, !(names(df) %in% c('debt_withdrawal', 'power_withdrawal'))]
npolicy <- length(unique(df$policy))
nyear <- dim(df)[1] / npolicy
df['id'] <- 1:nyear

# standardize
maxs <- apply(df, 2, max)
maxs['fund_hedge'] <- max(ceiling(maxs['fund_hedge']), ceiling(maxs['fund_withdrawal']))
maxs['fund_withdrawal'] <- max(ceiling(maxs['fund_hedge']), ceiling(maxs['fund_withdrawal']))
maxs['debt_hedge'] <- max(ceiling(maxs['debt_hedge']), ceiling(maxs['debt_withdrawal']))
maxs['debt_withdrawal'] <- max(ceiling(maxs['debt_hedge']), ceiling(maxs['debt_withdrawal']))
maxs['power_hedge'] <- max(ceiling(maxs['power_hedge']), ceiling(maxs['power_withdrawal']))
maxs['power_withdrawal'] <- max(ceiling(maxs['power_hedge']), ceiling(maxs['power_withdrawal']))
maxs['cash_in'] <- ceiling(maxs['cash_in'])
maxs['action_withdrawal'] <- ceiling(maxs['action_withdrawal'])
maxs['action_hedge'] <- ceiling(maxs['action_hedge']*10)/10
mins <- apply(df, 2, min)
mins['fund_hedge'] <- min(floor(mins['fund_hedge']), floor(mins['fund_withdrawal']))
mins['fund_withdrawal'] <- min(floor(mins['fund_hedge']), floor(mins['fund_withdrawal']))
mins['debt_hedge'] <- min(floor(mins['debt_hedge']), floor(mins['debt_withdrawal']))
mins['debt_withdrawal'] <- min(floor(mins['debt_hedge']), floor(mins['debt_withdrawal']))
mins['power_hedge'] <- min(floor(mins['power_hedge']), floor(mins['power_withdrawal']))
mins['power_withdrawal'] <- min(floor(mins['power_hedge']), floor(mins['power_withdrawal']))
mins['cash_in'] <- floor(mins['cash_in'])
mins['action_withdrawal'] <- floor(mins['action_withdrawal'])
mins['action_hedge'] <- floor(mins['action_hedge']*10)/10

# scale
print(df$policy)

#df_scale <- as.data.frame(t((t(df) - means) / stds))
df_scale_1 <- as.data.frame(t((t(df[df$policy == 1766, ]) - mins) / (maxs - mins)))
df_scale_2 <- as.data.frame(t((t(df[df$policy == 221, ]) - mins) / (maxs - mins)))
df_scale_3 <- as.data.frame(t((t(df[df$policy == 2002, ]) - mins) / (maxs - mins)))

# convert to long format (melt) & get hedge/withdrawal subsets
hedge_vars <- c('fund_hedge','debt_hedge','power_hedge','action_hedge')
withdrawal_vars <- c('fund_withdrawal','debt_withdrawal','power_withdrawal','cash_in','action_withdrawal')
df_long_1_h <- melt(df_scale_1, id.vars='id', measure.vars=hedge_vars)
df_long_2_h <- melt(df_scale_2, id.vars='id', measure.vars=hedge_vars)
df_long_3_h <- melt(df_scale_3, id.vars='id', measure.vars=hedge_vars)
df_long_1_w <- melt(df_scale_1, id.vars='id', measure.vars=withdrawal_vars)
df_long_2_w <- melt(df_scale_2, id.vars='id', measure.vars=withdrawal_vars)
df_long_3_w <- melt(df_scale_3, id.vars='id', measure.vars=withdrawal_vars)

# but get policy for each row, to use for color
df_match_1_h <- join(df_long_1_h, df_scale_1, by='id')
df_match_2_h <- join(df_long_2_h, df_scale_2, by='id')
df_match_3_h <- join(df_long_3_h, df_scale_3, by='id')
df_match_1_w <- join(df_long_1_w, df_scale_1, by='id')
df_match_2_w <- join(df_long_2_w, df_scale_2, by='id')
df_match_3_w <- join(df_long_3_w, df_scale_3, by='id')

# unnormalize for color
df_match_1_h$color <- df_match_1_h$action_hedge * (maxs['action_hedge'] - mins['action_hedge'])  + mins['action_hedge']
df_match_2_h$color <- df_match_2_h$action_hedge * (maxs['action_hedge'] - mins['action_hedge']) + mins['action_hedge']
df_match_3_h$color <- df_match_3_h$action_hedge * (maxs['action_hedge'] - mins['action_hedge']) + mins['action_hedge']
df_match_1_w$color <- df_match_1_w$action_withdrawal * (maxs['action_withdrawal'] - mins['action_withdrawal']) + mins['action_withdrawal']
df_match_2_w$color <- df_match_2_w$action_withdrawal * (maxs['action_withdrawal'] - mins['action_withdrawal']) + mins['action_withdrawal']
df_match_3_w$color <- df_match_3_w$action_withdrawal * (maxs['action_withdrawal'] - mins['action_withdrawal']) + mins['action_withdrawal']

# scales for axes
labels_hedge_before <- c('fund_hedge','debt_hedge','power_hedge','action_hedge')
labels_hedge_after <- c('Fund ($M)','Debt ($M)','Power ($/MWh)','Hedge ($M/inch)')
labels_withdrawal_before <- c('fund_withdrawal','debt_withdrawal','power_withdrawal','cash_in','action_withdrawal')
labels_withdrawal_after <- c('Fund ($M)','Debt ($M)','Power ($/MWh)','Cash ($M)','Hedge ($M/inch)')
    
df_match_1_h$variable <- mapvalues(df_match_1_h$variable, from=labels_hedge_before, to=labels_hedge_after)
df_match_2_h$variable <- mapvalues(df_match_2_h$variable, from=labels_hedge_before, to=labels_hedge_after)
df_match_3_h$variable <- mapvalues(df_match_3_h$variable, from=labels_hedge_before, to=labels_hedge_after)
df_match_1_w$variable <- mapvalues(df_match_1_w$variable, from=labels_withdrawal_before, to=labels_withdrawal_after)
df_match_2_w$variable <- mapvalues(df_match_2_w$variable, from=labels_withdrawal_before, to=labels_withdrawal_after)
df_match_3_w$variable <- mapvalues(df_match_3_w$variable, from=labels_withdrawal_before, to=labels_withdrawal_after)

p1 <- ggplot(data=df_match_1_h) +
    geom_line(aes(x=variable, y=value, group=id, color=color))+ #, alpha=0.3) +
    scale_colour_viridis_c(limits=c(mins['action_hedge'],maxs['action_hedge']), option='viridis', guide=F) + theme_bw() +
    theme(axis.text.x=element_blank(), axis.title.y=element_blank(),  axis.text.y=element_blank(),
        axis.ticks.x=element_blank(),axis.ticks.y=element_blank(), axis.title.x=element_blank(), panel.grid.major.y=element_blank(), panel.grid.minor.y=element_blank()) +
    geom_text(data=data.frame(x=labels_hedge_after,y=c(0,0,0,0),label=mins[c('fund_hedge','debt_hedge','power_hedge','action_hedge')]), 
        aes(x=x, y=y, label=label),size=2, color='black') + #,bg.colour='white') +
    geom_text(data=data.frame(x=labels_hedge_after,y=c(1,1,1,1),label=maxs[c('fund_hedge','debt_hedge','power_hedge','action_hedge')]), 
        aes(x=x, y=y, label=label),size=2, color='black')#,bg.colour='white')
ggsave(paste(dir_figs, "policy_hedgeA.eps", sep=''), p1, width=5, height=2.5, units='in')#, dpi=1000)

p2 <- ggplot(data=df_match_2_h) +
    geom_line(aes(x=variable, y=value, group=id, color=color))+ #, alpha=0.3) +
    scale_colour_viridis_c(limits=c(mins['action_hedge'],maxs['action_hedge']), option='viridis', guide=F) + theme_bw() +
    theme(axis.text.x=element_blank(), axis.title.y=element_blank(),  axis.text.y=element_blank(),
        axis.ticks.x=element_blank(),axis.ticks.y=element_blank(), axis.title.x=element_blank(), panel.grid.major.y=element_blank(), panel.grid.minor.y=element_blank()) +
    geom_text(data=data.frame(x=labels_hedge_after,y=c(0,0,0,0),label=mins[c('fund_hedge','debt_hedge','power_hedge','action_hedge')]), 
        aes(x=x, y=y, label=label),size=2, color='black')+ #,bg.colour='white') +
    geom_text(data=data.frame(x=labels_hedge_after,y=c(1,1,1,1),label=maxs[c('fund_hedge','debt_hedge','power_hedge','action_hedge')]), 
        aes(x=x, y=y, label=label),size=2, color='black')#,bg.colour='white') 
ggsave(paste(dir_figs, "policy_hedgeB.eps", sep=''), p2, width=5, height=2.5, units='in')#, dpi=1000)

p3 <- ggplot(data=df_match_3_h) +
    geom_line(aes(x=variable, y=value, group=id, color=color))+ #, alpha=0.3) +
    scale_colour_viridis_c(limits=c(mins['action_hedge'],maxs['action_hedge']), option='viridis', guide=F) + theme_bw() +
    theme(axis.text.x=element_blank(), axis.title.y=element_blank(),  axis.text.y=element_blank(),
        axis.ticks.x=element_blank(),axis.ticks.y=element_blank(), axis.title.x=element_blank(), panel.grid.major.y=element_blank(), panel.grid.minor.y=element_blank()) +
    geom_text(data=data.frame(x=labels_hedge_after,y=c(0,0,0,0),label=mins[c('fund_hedge','debt_hedge','power_hedge','action_hedge')]), 
        aes(x=x, y=y, label=label),size=2, color='black')+ #,bg.colour='white') +
    geom_text(data=data.frame(x=labels_hedge_after,y=c(1,1,1,1),label=maxs[c('fund_hedge','debt_hedge','power_hedge','action_hedge')]), 
        aes(x=x, y=y, label=label),size=2, color='black') #,bg.colour='white') 
ggsave(paste(dir_figs, "policy_hedgeC.eps", sep=''), p3, width=5, height=2.5, units='in')#, dpi=1000)




p4 <- ggplot(data=df_match_1_w) +
    geom_line(aes(x=variable, y=value, group=id, color=color))+ #, alpha=0.3) +
    scale_colour_viridis_c(limits=c(mins['action_withdrawal'],maxs['action_withdrawal']), option='viridis', guide=F) + theme_bw() +
    theme(axis.text.x=element_blank(), axis.title.y=element_blank(),  axis.text.y=element_blank(),
        axis.ticks.x=element_blank(),axis.ticks.y=element_blank(), axis.title.x=element_blank(), panel.grid.major.y=element_blank(), panel.grid.minor.y=element_blank()) +
    geom_text(data=data.frame(x=labels_withdrawal_after,y=c(0,0,0,0,0),label=mins[c('fund_withdrawal','debt_withdrawal','power_withdrawal','cash_in','action_withdrawal')]), 
        aes(x=x, y=y, label=label),size=2, color='black')+ #,bg.colour='white') +
    geom_text(data=data.frame(x=labels_withdrawal_after,y=c(1,1,1,1,1),label=maxs[c('fund_withdrawal','debt_withdrawal','power_withdrawal','cash_in','action_withdrawal')]), 
        aes(x=x, y=y, label=label),size=2, color='black')+ #,bg.colour='white') +
    geom_text(data=data.frame(x=labels_withdrawal_after[4:5],y=c((-mins['cash_in']/(maxs['cash_in']-mins['cash_in'])), (-mins['action_withdrawal']/(maxs['action_withdrawal']-mins['action_withdrawal']))),label=c(0,0)), 
        aes(x=x, y=y, label=label),size=2, color='black') #,bg.colour='white') 
ggsave(paste(dir_figs, "policy_withdrawalA.eps", sep=''), p4, width=5, height=2.5, units='in')#, dpi=1000)

p5 <- ggplot(data=df_match_2_w) +
    geom_line(aes(x=variable, y=value, group=id, color=color))+ #, alpha=0.3) +
    scale_colour_viridis_c(limits=c(mins['action_withdrawal'],maxs['action_withdrawal']), option='viridis', guide=F) + theme_bw() +
    theme(axis.text.x=element_blank(), axis.title.y=element_blank(),  axis.text.y=element_blank(),
        axis.ticks.x=element_blank(),axis.ticks.y=element_blank(), axis.title.x=element_blank(), panel.grid.major.y=element_blank(), panel.grid.minor.y=element_blank()) +
    geom_text(data=data.frame(x=labels_withdrawal_after,y=c(0,0,0,0,0),label=mins[c('fund_withdrawal','debt_withdrawal','power_withdrawal','cash_in','action_withdrawal')]), 
        aes(x=x, y=y, label=label),size=2, color='black')+ #,bg.colour='white') +
    geom_text(data=data.frame(x=labels_withdrawal_after,y=c(1,1,1,1,1),label=maxs[c('fund_withdrawal','debt_withdrawal','power_withdrawal','cash_in','action_withdrawal')]), 
        aes(x=x, y=y, label=label),size=2, color='black')+ #,bg.colour='white') +
    geom_text(data=data.frame(x=labels_withdrawal_after[4:5],y=c((-mins['cash_in']/(maxs['cash_in']-mins['cash_in'])), (-mins['action_withdrawal']/(maxs['action_withdrawal']-mins['action_withdrawal']))),label=c(0,0)), 
        aes(x=x, y=y, label=label),size=2, color='black') #,bg.colour='white') 
ggsave(paste(dir_figs, "policy_withdrawalB.eps", sep=''), p5, width=5, height=2.5, units='in')#, dpi=1000)

p6 <- ggplot(data=df_match_3_w) +
    geom_line(aes(x=variable, y=value, group=id, color=color))+ #, alpha=0.3) +
    scale_colour_viridis_c(limits=c(mins['action_withdrawal'],maxs['action_withdrawal']), option='viridis', guide=F) + theme_bw() +
    theme(axis.text.x=element_blank(), axis.title.y=element_blank(),  axis.text.y=element_blank(),
        axis.ticks.x=element_blank(),axis.ticks.y=element_blank(), axis.title.x=element_blank(), panel.grid.major.y=element_blank(), panel.grid.minor.y=element_blank()) +
    geom_text(data=data.frame(x=labels_withdrawal_after,y=c(0,0,0,0,0),label=mins[c('fund_withdrawal','debt_withdrawal','power_withdrawal','cash_in','action_withdrawal')]), 
        aes(x=x, y=y, label=label),size=2, color='black')+ #,bg.colour='white') +
    geom_text(data=data.frame(x=labels_withdrawal_after,y=c(1,1,1,1,1),label=maxs[c('fund_withdrawal','debt_withdrawal','power_withdrawal','cash_in','action_withdrawal')]), 
        aes(x=x, y=y, label=label),size=2, color='black')+ #,bg.colour='white') +
    geom_text(data=data.frame(x=labels_withdrawal_after[4:5],y=c((-mins['cash_in']/(maxs['cash_in']-mins['cash_in'])), (-mins['action_withdrawal']/(maxs['action_withdrawal']-mins['action_withdrawal']))),label=c(0,0)), 
        aes(x=x, y=y, label=label),size=2, color='black') #,bg.colour='white') 
ggsave(paste(dir_figs, "policy_withdrawalC.eps", sep=''), p6, width=5, height=2.5, units='in')#, dpi=1000)
# 
# g1 <- ggplotGrob(p1)
# g2 <- ggplotGrob(p2)
# g3 <- ggplotGrob(p3)
# g4 <- ggplotGrob(p4)
# g5 <- ggplotGrob(p5)
# g6 <- ggplotGrob(p6)
# g2$heights <- g1$heights
# g3$heights <- g1$heights
# g4$heights <- g1$heights
# g5$heights <- g1$heights
# g6$heights <- g1$heights
# grid.newpage()
# ptot <- grid.arrange(g1,g4,g2,g5,g3,g6, nrow=3)
# ggsave(paste(dir_figs,"policy_parallelCoord.png",sep=''), ptot, dpi=1000)



# solo versions for colorbar & labels
p7 <- ggplot(data=df_match_3_h) +
    geom_line(aes(x=variable, y=value, group=id, color=color))+ #, alpha=0.7) +
    scale_colour_viridis_c(limits=c(mins['action_hedge'],maxs['action_hedge']), breaks = c(0,0.25,0.5,0.75,1,1.25), option='viridis', name=NULL) + 
    theme_bw() +
    theme(axis.text.x=element_text(labels_hedge_after), axis.title.y=element_blank(),  axis.text.y=element_blank(),
        axis.ticks.x=element_blank(),axis.ticks.y=element_blank(), axis.title.x=element_blank(), panel.grid.major.y=element_blank(), panel.grid.minor.y=element_blank(), legend.position="bottom", legend.box = "horizontal") 
ggsave(paste(dir_figs, "policy_hedgeLeg.eps", sep=''), p7, width=5, height=2.5, units='in')#, dpi=1000)

p8 <- ggplot(data=df_match_3_w) +
    geom_line(aes(x=variable, y=value, group=id, color=color))+ #, alpha=0.7) +
    scale_colour_viridis_c(limits=c(mins['action_withdrawal'],maxs['action_withdrawal']), breaks = c(-30,-20,-10,0,10,20), option='viridis', name=NULL) + 
    theme_bw() +
    theme(axis.text.x=element_text(labels_withdrawal_after), axis.title.y=element_blank(),  axis.text.y=element_blank(),
        axis.ticks.x=element_blank(),axis.ticks.y=element_blank(), axis.title.x=element_blank(), panel.grid.major.y=element_blank(), panel.grid.minor.y=element_blank(), legend.position="bottom", legend.box = "horizontal") 
ggsave(paste(dir_figs, "policy_withdrawalLeg.eps", sep=''), p8, width=5, height=2.5, units='in')#, dpi=1000)

# g7 <- ggplotGrob(p7)
# g8 <- ggplotGrob(p8)
# g7$heights <- g8$heights
# grid.newpage()
# ptot2 <- grid.arrange(g7,g8, ncol=2)
# ggsave(paste(dir_figs, "policy_parallelCoord_legends.png", sep=''), ptot2, dpi=1000)








