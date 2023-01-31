# graspingObjectWithModelGenerated

estrarreNuvolaPuntiConCiclo.py -> Estrae la point cloud filtrando sull'asse z

export_ply.py -> Crea un file con formato PLY

opencv_pointcloud_viewer.py -> Visualizzazione a colori della finestra OpenCV, permette anche di interagire

opencv_viewer.py -> Depth visualization

union_pc.py -> Salva in unico file PLY due pointcloud che visualizza e le sovrappone mediante la funzione "union"

three_union_pc_with_save.py -> Salva tre point cloud (in file PLY) e le sovrappone mediante la funzione "union" 

iterarative_union_pc.py -> (con salvataggio) Può prendere in input al massimo 100 point cloud: la prima viene salvata in "prova_iterative.ply", la seconda viene salvata in "prova_iterative0.ply" e sovrapposta alla prima mediante la funzione "union"; la point cloud combinata viene salvata in "prova_iterative.ply" sovrascrivendo la prima point cloud acquisita. Questo procedimento viene eseguito per le successive iterazioni.


## LIBRERIE 
lib.py
lib_m.py


## PER LO STREAMING 
continue_visualization_of_pc.py -> Prima visualizzazione della point cloud in streaming

streaming.py -> Visualizzazione in streaming con salvataggio delle point cloud in 'model.py'

## RICOSTRUZIONE DELLA SCENA 
global_registration_mm.py -> Esegue la Global registration di point cloud salvate su file

multiway_registration.py -> Algoritmo utilizzato per la ricostruzione dell'oggetto nello spazio 3D. Dopo aver posizionato l'oggetto, è possibile salvare le point cloud che si desidera acquisire premendo "s" (le point cloud vengono salvate in una cartella rinominata "prova_iterative"). Per mostrare il modello finale premere "b": questo allineerà tutti i punti eseguendo la Multiway registration. Per chiudere l'applicazione premere "q".

## ROTAZIONI DEI MODELLI FINALI 
rotation_m.py -> Ruota una point cloud ("Modello_Finale.ply") di un angolo espresso in radianti intorno all'asse z e salva su file tutte le diverse rotazioni

rotation_and_moltiplication.py -> Ruota di 45° le point cloud intorono all'asse z e le salva. Qui viene definita la matrice contenente la posa (diversa per ogni tipo di oggetto). Scrive sul file di testo "File_matrix.txt" le coordinate relative alla posizione e le normali rispetto agli assi x, y e z di tutte e otto le rotazioni (su un'unica riga)

## PER LA PRESA
ply_multiway_registration_with_camera.py -> Sovrappone una point cloud acquisita dalla fotocamera con tutte e 8 le rotazioni di 45 gradi del modello

main_registration_pose.py -> Algoritmo utilizzato per il riconoscimento: mostra in output la fitness relativa alla nuova point cloud acquisita e il modello precedentemente ruotato intorno all'asse z. Inoltre, fornisce in output la matrice che serve per comandare il robot verso il punto di presa.
