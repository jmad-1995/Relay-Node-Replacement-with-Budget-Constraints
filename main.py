from matplotlib.pyplot import figure
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import math 
import heapq
import pylab
import itertools

G = nx.Graph()
G_mst=nx.Graph()


def create_graph(sensor_nodes,Comm_Range,Budget):
    global G
    global G_mst
    pos={}
    
    #Add nodes to the graph G
    for row in sensor_nodes.iterrows():
        G.add_node(row[1][0], x=row[1][1], y=row[1][2])
        G_mst.add_node(row[1][0], x=row[1][1], y=row[1][2])
        pos[row[1][0]]=[row[1][1],row[1][2]]
    for i in range(len(sensor_nodes)-1):
        x1=np.float64(G.nodes[sensor_nodes[0][i]]['x'])
        y1=np.float64(G.nodes[sensor_nodes[0][i]]['y'])
        node1=sensor_nodes[0][i]
        for j in range(i+1,len(sensor_nodes)):
            x2=np.float64(G.nodes[sensor_nodes[0][j]]['x'])
            y2=np.float64(G.nodes[sensor_nodes[0][j]]['y'])
            length=(((x1-x2)**2)+((y1-y2)**2))**0.5
            weight_node=(math.ceil(length/Comm_Range))-1
            node2=sensor_nodes[0][j]
            G.add_edge(node1,node2,weight=weight_node)
#----------------------------------------------------------------------
#Visualizing the graph
#----------------------------------------------------------------------
    fig=plt.figure(num=1, figsize=(15, 6), dpi=80, facecolor='w')
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    # edges
    nx.draw_networkx_edges(G, pos, width=6, alpha=0.5, edge_color='b', style='dashed')
    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    
    plt.axis('on')
    plt.title("Complete graph of sensor nodes")
    plt.show()

def create_spanning_tree(sensor_nodes,Comm_Range,Budget):
    #Create MST from the graph
    global G
    mst = defaultdict(set)
    starting_vertex=sensor_nodes[0][0]
    visited = set([starting_vertex])
    edges = [(weight, starting_vertex, to) for to, weight in G[starting_vertex].items()]
    counter = itertools.count()     # unique sequence count
    count=next(counter)
    edges_heap=[((edges[i][0]['weight']),next(counter),edges[i][1],edges[i][2]) for i in range(len(edges))]
    heapq.heapify(edges_heap)
    while edges_heap:
        weight, count_present,frm, to = heapq.heappop(edges_heap)
        if to not in visited:
            visited.add(to)
            mst[frm].add(to)
            next_node_dict=[(to_next,weight) for to_next,weight in G[to].items()]
            next_node_list=[((next_node_dict[i][1]['weight']),next_node_dict[i][0]) for i in range(len(next_node_dict))]
            
            for weight,to_next in next_node_list:
                if to_next not in visited:
                    heapq.heappush(edges_heap, (weight,next(counter), to, to_next))
#                    print("edges_heap after push:",edges_heap)
            heapq.heapify(edges_heap)
#    print("visited final:",visited)
    #print("MST final:",mst)
#    
    return mst
    
def MST_Package():
    #Using networkx minimum spanning tree using prim's algorithm to check if
    #we are getting the same MST
    global G
    T=nx.minimum_spanning_tree(G,weight='weight',algorithm='prim')
    #print("Package MST:")
    #print(sorted(T.edges(data=True)))
    #print("done")
    
def BCRP_MNCC(mst,sensor_nodes,Comm_Range,Budget):
    global G
    global G_mst
    pos={} 
    Rem_nodes=0
    Conn_Compo=1
    #Representing the actual MST
    #Nodes are already added 
    #in G_mst but there are no edges.based on MST we,get,we will add edges in G_mst.
    #So,G_mst will be our final MST
    for row in sensor_nodes.iterrows():
        get_set=mst.get(row[1][0])
        #print( "get the first element of mst dictionary:",get_set)
        if get_set is not None:
            for i in range(len(get_set)):
                next_elem=get_set.pop()
                weight=G[row[1][0]][next_elem]['weight']                
                #print(weight)
                G_mst.add_edge(row[1][0],next_elem,weight=weight)
            
#------------------------------------------------------------------------------        
#Visualization
#------------------------------------------------------------------------------  
    for row in sensor_nodes.iterrows():
        pos[row[1][0]]=[row[1][1],row[1][2]]
    fig=plt.figure(num=1, figsize=(15, 6), dpi=80, facecolor='w')
    # nodes
    nx.draw_networkx_nodes(G_mst, pos, node_size=400)

    # edges
    nx.draw_networkx_edges(G_mst, pos, edge_color='k', width=4)

    # labels
    nx.draw_networkx_labels(G_mst, pos, font_size=20, font_family='sans-serif')

    
    plt.axis('on')
    plt.title("MST graph of Sensor nodes")
    plt.show()
    #print("MST of my graph:")
    #print(G_mst.nodes(data=True))
    #print(G_mst.edges.data())
    #print(type(G_mst))
    #print("Total weight:",G_mst.size(weight='weight'))

##-------------------------------------------------------------------------------------
##Algortihm 4 Implemenation
##-------------------------------------------------------------------------------------
    if (G_mst.size(weight='weight')<= Budget):
         #print("Number of connected components:",Conn_Compo)
#--------------------------------------------------------------------------
#Visualization
#--------------------------------------------------------------------------
         for row in sensor_nodes.iterrows():
              pos[row[1][0]]=[row[1][1],row[1][2]]
         fig=plt.figure(num=1, figsize=(15, 6), dpi=80, facecolor='w')
         # nodes
         nx.draw_networkx_nodes(G_mst, pos, node_size=400)

         # edges
         nx.draw_networkx_edges(G_mst, pos, edge_color='k', width=4)

         # labels
         nx.draw_networkx_labels(G_mst, pos, font_size=20, font_family='sans-serif')
         
         plt.axis('on')
         plt.title("MST graph of Sensor nodes")
         plt.show()
         
    else: 
         while (G_mst.size(weight='weight')> Budget):
            a,b,data=sorted(G_mst.edges(data=True), key=lambda x: x[2]['weight'],reverse=True)[0]
            #print('{a} {b} {w}'.format(a=a, b=b, w=data['weight']))
            G_mst.remove_edge(a,b)
            Rem_nodes+=1
            Conn_Compo+=1
            #print("G_mst after removal of ",Rem_nodes," maximum weighted edges:",G_mst.edges.data())
            #print("Number of connected components:",Conn_Compo)
#--------------------------------------------------------------------------
#Visualization
#--------------------------------------------------------------------------
            for row in sensor_nodes.iterrows():
                pos[row[1][0]]=[row[1][1],row[1][2]]
            fig=plt.figure(num=1, figsize=(15, 6), dpi=80, facecolor='w')
            # nodes
            nx.draw_networkx_nodes(G_mst, pos, node_size=400)

            # edges
            nx.draw_networkx_edges(G_mst, pos, edge_color='k', width=4)

            # labels
            nx.draw_networkx_labels(G_mst, pos, font_size=20, font_family='sans-serif')
            
            plt.axis('on')
            plt.title("MST graph of Sensor nodes after removal of {nodes} maximum weighted edges ".format(nodes=Rem_nodes))
            plt.show()
            

def create_k_spanning_tree(mst):
  leaf_nodes=find_leaf_nodes(mst)
  max_tuple=max(leaf_nodes, key=lambda x: x[1])
  for node in mst[max_tuple[0]]:
    mst[node].remove(max_tuple[0])
  del mst[max_tuple[0]]
  return mst,max_tuple[1]
          
def genAdjlst(graphInput):
  print("generating adjacency list")
  newdict = graphInput.copy()
  for x in graphInput:
    for y in graphInput[x]:
        #newdict[x] = thisdict[x]
        if y in graphInput:
            newdict[y].add(x)
            continue
        newdict[y] = {x}
  return newdict
 
def sum_of_weights(mst):
  print("jmad")
  sum=0
  for node in mst:
    for connected_node in mst[node]:
      sum+=G.get_edge_data(node,connected_node)['weight']
  sum=sum/2;
  return sum;

def find_leaf_nodes(mst):
  leaf_nodes=[]  
  for node in mst:
    if len(mst[node])==1:
       for connected_node in mst[node]: 
          weight=G.get_edge_data(node,connected_node)['weight']
          leaf_nodes.append([node,weight])
  return leaf_nodes

def BRCP_MLCC(Comm_Range,Budget):
  print("Entering algo 5: BRCC_MLCC")
  sensor_nodes=pd.read_csv(r"sensor_location.csv",header=None)
  create_graph(sensor_nodes,Comm_Range, Budget)
  mst= genAdjlst(create_spanning_tree(sensor_nodes,Comm_Range,Budget))
  weights_sum=sum_of_weights(mst)
  n=len(sensor_nodes)
  print(n)
  visualise_my_mst(mst)
  for i in range(n,2,-1):
      if weights_sum <= Budget:
        visualise_my_mst(mst)
        return mst
      else:
        new_mst,deleted_node_weight= create_k_spanning_tree(mst)
        mst=new_mst
        weights_sum-=deleted_node_weight

def drawGraph(g):

  pos=nx.spring_layout(g)

  # version 2
  pylab.figure(2)
  nx.draw(g,pos,with_labels = True)
  # specifiy edge labels explicitly
  edge_labels=dict([((u,v,),d['weight'])
               for u,v,d in g.edges(data=True)])
  nx.draw_networkx_edge_labels(g,pos,edge_labels=edge_labels)


  # show graphs
  pylab.show()
  
def visualise_my_mst(mst):
  print(mst)
  global G
  new_graph = nx.Graph()
  for node in mst:
    new_graph.add_node(node)
  for node in mst:
    for connected_node in mst[node]:
      weight=G.get_edge_data(node,connected_node)['weight']
      new_graph.add_edge(node,connected_node,weight=weight)
  drawGraph(new_graph)

      
      
if __name__=='__main__':
    global R
    global B
    #When this program is run,it asks the user to give coomunication range and budget as 
    #input.It will read the csv file where the nodes and their locations are mentioned.
    #Manipulation of number of nodes can be done in csv file for analysis purpose.
    R=float(input("Enter communication range:"))
    B=float(input("Enter the relay nodes budget(Please provide integer):"))
    #Reading from csv file
    sensor_nodes=pd.read_csv("sensor_location.csv",header=None)
    #To create complete graph
    #create_graph(sensor_nodes,R,B)
    #To create MST out of the complete graph
    #mst=create_spanning_tree(sensor_nodes,R,B)
    #print(genAdjlst(mst))
    #print("MST for BCRP-MNCC Algorithm 4:",mst)
    #This function is called to confirm if we are getting the same MST as the MST generated by package 
    #present in networkx
    #MST_Package()
    #Implementation of Algorithm 4
    #BCRP_MNCC(mst,sensor_nodes,R,B)
    #Implementation of Algorithm 5
    BRCP_MLCC(R,B)
