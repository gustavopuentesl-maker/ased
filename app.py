import streamlit as st
import os, re, warnings, requests, base64, json
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─── Configuración página ─────────────────────────────────
st.set_page_config(page_title="Clasificador Fallas CEC",page_icon="⚡",layout="wide")
st.markdown("""<style>
.resultado-FM{border-left:5px solid #e74c3c;background:#fdf0ef;border-radius:8px;padding:14px;margin:10px 0}
.resultado-E{border-left:5px solid #f39c12;background:#fef9ef;border-radius:8px;padding:14px;margin:10px 0}
.resultado-I{border-left:5px solid #27ae60;background:#edfaf1;border-radius:8px;padding:14px;margin:10px 0}
.badge-FM{background:#e74c3c;color:white;padding:4px 12px;border-radius:20px;font-weight:bold}
.badge-E{background:#f39c12;color:white;padding:4px 12px;border-radius:20px;font-weight:bold}
.badge-I{background:#27ae60;color:white;padding:4px 12px;border-radius:20px;font-weight:bold}
.geo-card{background:#eaf4fb;border:2px solid #3498db;border-radius:8px;padding:14px;margin:10px 0}
.sec{font-size:16px;font-weight:bold;color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:4px;margin-bottom:10px}
</style>""",unsafe_allow_html=True)

# ─── Configuración GitHub ─────────────────────────────────
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO  = st.secrets["GITHUB_REPO"]
ARCHIVO_CSV  = "fallas_ingresadas.csv"
EXCEL_PATH   = "Estadisticas_de_Interrupciones.xlsx"

# ─── Constantes ───────────────────────────────────────────
ETIQUETAS    = {"FM":"🔴 Fuerza Mayor","E":"🟡 Externa","I":"🟢 Interna"}
COLORES      = {"FM":"#e74c3c","E":"#f39c12","I":"#27ae60"}
COLORES_ALIM = {2:"blue",6:"red",7:"green",8:"purple",9:"orange",10:"darkblue"}

CUADRILLAS = [
    "Cuadrilla 1 — Curicó Norte","Cuadrilla 2 — Curicó Sur",
    "Cuadrilla 3 — Teno","Cuadrilla 4 — Molina",
    "Cuadrilla 5 — Romeral","Cuadrilla 6 — Chimbarongo",
    "Cuadrilla 7 — Emergencia","Otra (escribir abajo)",
]

CAUSAS = {
    "🌿 Árboles / Vegetación":{"cal":"I","subs":[
        ("Caída de árbol sobre la red","Caída de árbol sobre red MT/BT"),
        ("Caída de gancho o rama","Caída de rama o gancho sobre conductor"),
        ("Poda inadecuada por cliente","Cliente efectuó poda propia indebida afectando red"),
        ("Poda inadecuada por municipalidad","Municipalidad efectuó poda que dañó la red"),
        ("Sector sin podar","Sector sin podar o mal podado afecta red"),
        ("Daño por empresa forestal","Daño por faena de empresa forestal en franja"),
        ("Otro árbol/vegetación","Daño por vegetación — causa no especificada")]},
    "⛈ Condiciones Atmosféricas":{"cal":"I","subs":[
        ("Temporal / viento fuerte","Interrupción por temporal o viento fuerte"),
        ("Lluvia intensa","Interrupción por lluvia intensa"),
        ("Granizo","Interrupción por granizo"),
        ("Nieve","Interrupción por nieve"),
        ("Hielo en conductores","Interrupción por hielo en conductores"),
        ("Rayo / descarga eléctrica","Descarga eléctrica (rayo) impacta instalación"),
        ("Neblina / ambiente salino","Interrupción por neblina o ambiente salino"),
        ("Temperatura extrema","Interrupción por temperatura extrema")]},
    "🚗 Accidentes / Vehículos":{"cal":"FM","subs":[
        ("Choque de vehículo a poste","Choque de vehículo a poste de distribución"),
        ("Vehículo alto bota cable MT","Vehículo > 4.5m bota cable media tensión"),
        ("Vehículo alto bota cable BT","Vehículo > 4.5m bota cable baja tensión"),
        ("Choque a tirante / retenida","Choque de vehículo a tirante o retenida"),
        ("Máquina retroexcavadora","Daño por máquina retroexcavadora"),
        ("Daño en faena particular","Daño debido a faena en propiedad particular"),
        ("Otro accidente","Accidente — causa no especificada")]},
    "🔥 Incendio":{"cal":"FM","subs":[
        ("Calor por incendio externo","Calor en instalación por incendio externo"),
        ("Quema de pastizales","Interrupción por quema de pastizales"),
        ("Solicitud de bomberos","Intervención a solicitud de bomberos"),
        ("Otro incendio","Incendio — causa no especificada")]},
    "🌊 Eventos Catastróficos":{"cal":"FM","subs":[
        ("Inundación","Interrupción por inundación"),
        ("Aluvión / deslizamiento","Interrupción por aluvión o deslizamiento"),
        ("Movimiento telúrico (sismo)","Interrupción por sismo"),
        ("Otro catastrófico","Evento catastrófico — no especificado")]},
    "🦅 Animales":{"cal":"I","subs":[
        ("Ave en instalación","Contacto de ave con instalación eléctrica"),
        ("Mamífero en instalación","Contacto de mamífero con instalación"),
        ("Roedor daña cable","Roedor daña aislación de cable"),
        ("Insecto / colmena","Insecto o colmena en instalación"),
        ("Otro animal","Animal — causa no especificada")]},
    "🔧 Falla en la Red (Interna)":{"cal":"I","subs":[
        ("Operación imprevista de equipo","Operación imprevista de equipo de maniobra"),
        ("Error de operación","Error de operación por personal"),
        ("Falla de material / equipo","Falla de material o equipo de la red"),
        ("Envejecimiento de materiales","Falla por envejecimiento de materiales"),
        ("Pérdida de aislación","Pérdida de aislación en conductor o equipo"),
        ("Sobrecarga / desequilibrio","Interrupción por sobrecarga"),
        ("Falla transformador distribución","Falla en transformador de distribución"),
        ("Mantenimiento preventivo","Corte programado por mantenimiento"),
        ("Otra falla interna","Falla interna — no especificada")]},
    "⚡ Causa Externa (Sistema)":{"cal":"E","subs":[
        ("Blackout del sistema","Blackout del sistema eléctrico nacional"),
        ("Falla en transmisión (Transelec)","Falla en línea/subestación de transmisión"),
        ("Falla en subtransmisión","Falla en línea/subestación de subtransmisión"),
        ("Racionamiento con decreto","Racionamiento energético con decreto vigente"),
        ("Baja/sobre frecuencia","Protección por baja o sobre frecuencia"),
        ("Programada generación","Interrupción programada por generación"),
        ("Otra causa externa","Causa externa — no especificada")]},
    "🎯 Actos Vandálicos":{"cal":"FM","subs":[
        ("Robo de conductor","Robo de conductor en red"),
        ("Robo de equipo","Robo de equipo eléctrico"),
        ("Objeto lanzado","Objeto lanzado sobre red"),
        ("Atentado / explosivos","Atentado con explosivos"),
        ("Sabotaje","Sabotaje a instalación eléctrica"),
        ("Otro vandálico","Acto vandálico — no especificado")]},
    "👤 Instalación de Cliente":{"cal":"I","subs":[
        ("Falla en artefacto eléctrico","Falla en artefacto eléctrico del cliente"),
        ("Problema neutro / puesta tierra","Problema de neutro o puesta a tierra"),
        ("Capacidad insuficiente S/E","Capacidad insuficiente subestación cliente"),
        ("Desconexión solicitada","Desconexión a solicitud del cliente"),
        ("Otro cliente","Instalación cliente — no especificada")]},
}

# ─── Funciones GitHub ─────────────────────────────────────
def cargar_hist():
    url     = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{ARCHIVO_CSV}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    r       = requests.get(url, headers=headers)
    if r.status_code == 200:
        contenido = base64.b64decode(r.json()["content"]).decode("utf-8")
        return pd.read_csv(StringIO(contenido))
    return pd.DataFrame()

def guardar(fila):
    dh  = cargar_hist()
    n   = len(dh) + 1
    fila["N°"]         = n
    fila["Fecha_hora"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dh  = pd.concat([dh, pd.DataFrame([fila])], ignore_index=True)
    csv_b64 = base64.b64encode(dh.to_csv(index=False).encode()).decode()
    url     = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{ARCHIVO_CSV}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}","Content-Type":"application/json"}
    r       = requests.get(url, headers=headers)
    sha     = r.json().get("sha") if r.status_code == 200 else None
    payload = {"message":f"Registro N°{n} — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
               "content":csv_b64}
    if sha: payload["sha"] = sha
    requests.put(url, headers=headers, data=json.dumps(payload))
    return n

# ─── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    col_l1,col_l2=st.columns(2)
    with col_l1:
        if os.path.exists("logo_udec.png"):
            st.image("logo_udec.png",width=100)
    with col_l2:
        if os.path.exists("logo_cec.png"):
            st.image("logo_cec.png",width=100)
    st.markdown("---")
    st.markdown("# ⚡ CEC Fallas")
    st.markdown("🌐 [Sitio web CEC](https://cecltda.cl/)")
    st.markdown("---")
    if not os.path.exists(EXCEL_PATH):
        st.error("❌ Excel no encontrado"); st.stop()
    else:
        st.success("✅ Sistema conectado")
    st.markdown("---")
    pagina=st.radio("Navegación:",[
        "📝 Registrar Falla","📋 Historial","🗺 Mapa General","ℹ️ Ayuda"])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:15px;color:#888;text-align:center;padding:10px 0'>
        <b>Desarrollado por:</b><br><br>
        <a href='https://www.linkedin.com/in/gustavo-puentes-lermanda-78830a25a'
           target='_blank' style='color:#0077b5;text-decoration:none'>
           👤 Gustavo Puentes Lermanda</a><br><br>
        <a href='https://www.linkedin.com/in/sofia-eliana-nahuelpán-álvarez-62b11a351'
           target='_blank' style='color:#0077b5;text-decoration:none'>
           👤 Sofía Nahuelpán Álvarez</a><br><br>
        <a href='https://www.linkedin.com/in/PERFIL-SEBASTIAN'
           target='_blank' style='color:#0077b5;text-decoration:none'>
           👤 Sebastián Castro Astudillo</a><br><br>
        <i>Depto. Ingeniería Eléctrica<br>Universidad de Concepción</i>
    </div>
    """,unsafe_allow_html=True)
    st.caption("Clasificador CEC v3 © 2026")

# ─── Cargar modelos ───────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Cargando datos y entrenando modelos...")
def cargar_modelos(path):
    def tf(v):
        try: return float(str(v).replace(",","."))
        except: return np.nan
    xl  = pd.read_excel(path,sheet_name=None)
    raw = xl["BD"].iloc[1:].copy()
    raw.columns=["_","INC","Alimentador","Trafos","KVA","Clientes","Inicio","Fin",
                 "Duracion","Causa","Calificacion","Comuna_id","Comuna","Tension",
                 "X","Y","Descripcion","Validacion","año","Falta_coord"]
    raw["X_f"]=raw["X"].apply(tf); raw["Y_f"]=raw["Y"].apply(tf)
    raw.loc[(raw["X_f"]<280000)|(raw["X_f"]>380000),"X_f"]=np.nan
    raw.loc[(raw["Y_f"]<6080000)|(raw["Y_f"]>6160000),"Y_f"]=np.nan
    df=raw[["INC","Alimentador","Calificacion","Descripcion","año","X_f","Y_f","Comuna"]].copy()
    df["Calificacion"]=df["Calificacion"].str.strip().str.upper()
    df=df[df["Calificacion"].isin(["I","FM","E"])]
    u2w=Transformer.from_crs("EPSG:32719","EPSG:4326",always_xy=True)
    def u2l(x,y): lo,la=u2w.transform(x,y); return la,lo
    dg=df.dropna(subset=["X_f","Y_f","Alimentador"]).copy()
    dg["Alimentador"]=dg["Alimentador"].astype(int)
    knn=KNeighborsClassifier(n_neighbors=5,weights="distance")
    knn.fit(dg[["X_f","Y_f"]].values,dg["Alimentador"].values)
    info={}
    for alim,grp in dg.groupby("Alimentador"):
        xc,yc=grp["X_f"].mean(),grp["Y_f"].mean(); la,lo=u2l(xc,yc)
        pts=[u2l(r.X_f,r.Y_f) for _,r in grp.iterrows()]
        info[int(alim)]={"x_utm":xc,"y_utm":yc,"lat":la,"lon":lo,
            "comunas":", ".join(sorted(set(grp["Comuna"].dropna().astype(str)))),
            "n_fallas":len(grp),"pts":pts}
    STOP={"de","la","el","en","y","a","los","las","por","con","se","del","al",
          "un","una","es","que","no","le","su","sus","para","como","fue",
          "pero","ha","hay","lo","me","si","o","e","u"}
    def limpiar(t):
        if not isinstance(t,str): return ""
        t=t.lower()
        for a,b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ñ","n")]:
            t=t.replace(a,b)
        t=re.sub(r"[^a-z0-9\s]"," ",t)
        return " ".join(x for x in t.split() if x not in STOP and len(x)>2)
    dm=df.dropna(subset=["Descripcion","Calificacion"]).copy()
    dm["txt"]=dm["Descripcion"].apply(limpiar)
    modelo=Pipeline([
        ("tfidf",TfidfVectorizer(ngram_range=(1,2),max_features=5000,min_df=2)),
        ("clf",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42))])
    modelo.fit(dm["txt"],dm["Calificacion"])
    return modelo,knn,info,limpiar,u2w

modelo,knn_alim,info_alim,limpiar,u2w=cargar_modelos(EXCEL_PATH)
def u2l(x,y): lo,la=u2w.transform(x,y); return la,lo

# ─── Funciones core ───────────────────────────────────────
def predecir(desc):
    tl=limpiar(desc); pm=modelo.predict([tl])[0]
    pr=dict(zip(modelo.classes_,modelo.predict_proba([tl])[0]))
    return {"pred":pm,"conf":round(pr[pm]*100,1),
            "FM":round(pr.get("FM",0)*100,1),
            "E":round(pr.get("E",0)*100,1),
            "I":round(pr.get("I",0)*100,1)}

def id_alim(x,y):
    if not x or not y or x<1000: return None
    p=np.array([[x,y]]); alim=int(knn_alim.predict(p)[0])
    proba=dict(zip(knn_alim.classes_,knn_alim.predict_proba(p)[0]))
    dist=float(knn_alim.kneighbors(p,n_neighbors=1)[0][0][0])
    c=info_alim[alim]; dc=np.sqrt((x-c["x_utm"])**2+(y-c["y_utm"])**2)
    return {"alim":alim,"conf":round(proba[alim]*100,1),"dist_p":round(dist,1),
            "dist_c":round(dc,1),"comunas":c["comunas"],
            "votos":{int(a):round(p*100,1) for a,p in proba.items() if p>0.01}}

# ══════════════════════════════════════════════════════════
# PÁGINA 1 — REGISTRAR FALLA
# ══════════════════════════════════════════════════════════
if pagina=="📝 Registrar Falla":
    st.markdown("## 📝 Registro de Interrupción")
    st.markdown('<div class="sec">👤 Sección 1 — Operador</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: usuario=st.text_input("Nombre del operador *",placeholder="Ej: Juan Pérez")
    with c2: cuad_sel=st.selectbox("Cuadrilla *",CUADRILLAS)
    with c3:
        if "Otra" in cuad_sel:
            cuadrilla=st.text_input("Nombre de cuadrilla:")
        else:
            cuadrilla=cuad_sel
            st.text_input("Cuadrilla:",value=cuadrilla,disabled=True)
    st.markdown("---")
    st.markdown('<div class="sec">📝 Sección 2 — Descripción de la falla</div>',unsafe_allow_html=True)
    modo=st.radio("Modo:",["🧭 Guiado (recomendado)","✏️ Manual"],horizontal=True)
    desc=""; cat=""; sub=""; cal_sug=None
    if "Guiado" in modo:
        ca,cb=st.columns(2)
        with ca: categoria=st.selectbox("Categoría:",list(CAUSAS.keys()))
        with cb:
            subs_opts=[s[0] for s in CAUSAS[categoria]["subs"]]
            subcausa=st.selectbox("Subcausa:",subs_opts)
        base=next((d for n,d in CAUSAS[categoria]["subs"] if n==subcausa),subcausa)
        det=st.text_area("Detalles adicionales:",placeholder="Sector, equipo, observaciones...",height=70)
        desc=base+(f". {det}" if det.strip() else "")
        cat=categoria; sub=subcausa; cal_sug=CAUSAS[categoria]["cal"]
        st.info(f"📋 Descripción generada: {desc}")
    else:
        desc=st.text_area("Descripción:",placeholder="Ej: Se abre reconectador RCU5724...",height=100)
        cat="Manual"; sub="Manual"
    st.markdown("---")
    st.markdown('<div class="sec">📍 Sección 3 — Coordenadas UTM zona 19S (opcional)</div>',unsafe_allow_html=True)
    cx,cy,ci=st.columns([2,2,3])
    with cx:
        xv_str=st.text_input("X — Este:",placeholder="Ej: 315569.5",key="coord_x")
        try: xv=float(xv_str.replace(",",".")) if xv_str.strip() else 0.0
        except: xv=0.0
    with cy:
        yv_str=st.text_input("Y — Norte:",placeholder="Ej: 6136152.3",key="coord_y")
        try: yv=float(yv_str.replace(",",".")) if yv_str.strip() else 0.0
        except: yv=0.0
    with ci:
        if xv>1000 and yv>1000:
            lp,lop=u2l(xv,yv); st.success(f"Lat:{lp:.5f} | Lon:{lop:.5f}")
        else: st.info("Ingresa X e Y para identificar el alimentador")
    st.markdown("---")
    if st.button("🔍 Clasificar falla",type="primary"):
        errs=[]
        if not usuario.strip(): errs.append("⚠️ Ingresa el nombre del operador.")
        if not desc.strip(): errs.append("⚠️ Ingresa o selecciona una descripción.")
        for e in errs: st.error(e)
        if not errs:
            rf=predecir(desc)
            if cal_sug and rf["conf"]<60: rf["pred"]=cal_sug
            x_coord=float(xv) if xv>1000 else None
            y_coord=float(yv) if yv>1000 else None
            rg=id_alim(x_coord,y_coord)
            st.session_state["ultimo"]={"usuario":usuario,"cuadrilla":cuadrilla,
                "desc":desc,"cat":cat,"sub":sub,"modo":modo,
                "rf":rf,"rg":rg,"x":x_coord,"y":y_coord}
            st.session_state["mostrar"]=True
    if st.session_state.get("mostrar") and st.session_state.get("ultimo"):
        u=st.session_state["ultimo"]; rf=u["rf"]; rg=u["rg"]
        x_coord=u["x"]; y_coord=u["y"]
        st.markdown(f"""<div class="resultado-{rf["pred"]}">
          <span class="badge-{rf["pred"]}">{ETIQUETAS[rf["pred"]]}</span>
          &nbsp;<span style="font-size:13px;color:#555">
            Confianza:{rf["conf"]}% | Operador:{u["usuario"]} | Cuadrilla:{u["cuadrilla"]}
          </span>
          <p style="margin:8px 0 0;font-size:13px"><b>Descripción:</b> {u["desc"]}</p>
        </div>""",unsafe_allow_html=True)
        p1,p2,p3=st.columns(3)
        p1.metric("🔴 FM",f"{rf[chr(70)+chr(77)]}%"); p1.progress(rf["FM"]/100)
        p2.metric("🟡 Externa",f"{rf[chr(69)]}%"); p2.progress(rf["E"]/100)
        p3.metric("🟢 Interna",f"{rf[chr(73)]}%"); p3.progress(rf["I"]/100)
        if rg:
            nivel="🟢 Alta" if rg["conf"]>=80 else "🟡 Media" if rg["conf"]>=50 else "🔴 Baja"
            votos=" | ".join(f"Alim.{a}:{p}%" for a,p in sorted(rg["votos"].items(),key=lambda x:-x[1]))
            st.markdown(f"""<div class="geo-card">
              <b>📍 Alimentador identificado: #{rg["alim"]}</b> {nivel} ({rg["conf"]}%)<br>
              Dist.punto histórico:{rg["dist_p"]}m | Comunas:{rg["comunas"]}<br>
              <small>Votos KNN: {votos}</small>
            </div>""",unsafe_allow_html=True)
            with st.expander("🗺 Ver mapa",expanded=True):
                m=folium.Map(location=[-34.98,-71.08],zoom_start=10)
                for alim,c in sorted(info_alim.items()):
                    col=COLORES_ALIM.get(alim,"gray"); es=rg["alim"]==alim
                    paso=max(1,len(c["pts"])//60)
                    for i in range(0,len(c["pts"]),paso):
                        folium.CircleMarker(c["pts"][i],radius=3,color=col,
                            fill=True,fill_opacity=0.35,weight=0).add_to(m)
                    folium.Marker([c["lat"],c["lon"]],tooltip=f"Alimentador {alim}",
                        popup=folium.Popup(f"<b>Alim.{alim}</b><br>Comunas:{c[chr(99)+chr(111)+chr(109)+chr(117)+chr(110)+chr(97)+chr(115)]}",max_width=180),
                        icon=folium.Icon(color="red" if es else col,
                            icon="bolt" if es else "info-sign",
                            prefix="fa" if es else "glyphicon")).add_to(m)
                if x_coord and y_coord:
                    lp,lop=u2l(x_coord,y_coord)
                    folium.Marker([lp,lop],tooltip="📍 Punto de la falla",
                        popup=f"X:{x_coord:.0f} Y:{y_coord:.0f}",
                        icon=folium.Icon(color="black",icon="map-marker",prefix="fa")).add_to(m)
                    folium.PolyLine([[lp,lop],[info_alim[rg["alim"]]["lat"],info_alim[rg["alim"]]["lon"]]],
                        color="red",weight=3,dash_array="6").add_to(m)
                    m.location=[lp,lop]; m.zoom_start=13
                st_folium(m,width=700,height=400)
        else:
            st.info("📍 Sin coordenadas — no se identificó alimentador.")
        st.markdown("---")
        if st.button("💾 Guardar registro",type="primary"):
            with st.spinner("Guardando en GitHub..."):
                n=guardar({
                    "Usuario":u["usuario"],"Cuadrilla":u["cuadrilla"],
                    "Descripcion":u["desc"],"Categoria":u["cat"],"Subcausa":u["sub"],
                    "Modo":"Guiado" if "Guiado" in u["modo"] else "Manual",
                    "Clasificacion":rf["pred"],
                    "Tipo_Falla":ETIQUETAS[rf["pred"]].split(" ",1)[1],
                    "Confianza_%":rf["conf"],"Prob_FM":rf["FM"],"Prob_E":rf["E"],"Prob_I":rf["I"],
                    "X_UTM":x_coord or "","Y_UTM":y_coord or "",
                    "Alimentador":rg["alim"] if rg else "",
                    "Confianza_Alim_%":rg["conf"] if rg else "",
                    "Dist_punto_m":rg["dist_p"] if rg else "",
                    "Comunas":rg["comunas"] if rg else ""})
            st.success(f"✅ Registro N°{n} guardado en GitHub → {ARCHIVO_CSV}")
            st.session_state["mostrar"]=False
            st.session_state["ultimo"]={}

# ══════════════════════════════════════════════════════════
# PÁGINA 2 — HISTORIAL
# ══════════════════════════════════════════════════════════
elif pagina=="📋 Historial":
    st.markdown("## 📋 Historial de Fallas")
    with st.spinner("Cargando historial desde GitHub..."):
        dh=cargar_hist()
    if dh.empty:
        st.info("📭 Sin registros aún.")
    else:
        c1,c2,c3,c4=st.columns(4); c1.metric("Total",len(dh))
        if "Clasificacion" in dh.columns:
            cnt=dh["Clasificacion"].value_counts()
            c2.metric("🔴 FM",cnt.get("FM",0))
            c3.metric("🟡 E",cnt.get("E",0))
            c4.metric("🟢 I",cnt.get("I",0))
        st.markdown("---")
        fa,fb=st.columns(2)
        with fa:
            if "Clasificacion" in dh.columns:
                ft=st.multiselect("Filtrar tipo:",["FM","E","I"],default=["FM","E","I"])
                dh=dh[dh["Clasificacion"].isin(ft)]
        with fb:
            if "Usuario" in dh.columns:
                us=["Todos"]+sorted(dh["Usuario"].dropna().unique().tolist())
                fu=st.selectbox("Filtrar usuario:",us)
                if fu!="Todos": dh=dh[dh["Usuario"]==fu]
        st.dataframe(dh,use_container_width=True,height=400)
        col_d1,col_d2=st.columns(2)
        with col_d1:
            csv=dh.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar CSV",csv,"historial_fallas.csv","text/csv")
        with col_d2:
            excel_buf=__import__("io").BytesIO()
            dh.to_excel(excel_buf,index=False); excel_buf.seek(0)
            st.download_button("⬇️ Descargar Excel",excel_buf,
                "historial_fallas.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════
# PÁGINA 3 — MAPA GENERAL
# ══════════════════════════════════════════════════════════
elif pagina=="🗺 Mapa General":
    st.markdown("## 🗺 Mapa de Alimentadores CEC")
    ma,mb=st.columns([3,1])
    with mb:
        st.markdown("**Alimentadores:**")
        for alim,c in sorted(info_alim.items()):
            st.markdown(f"● **Alim.{alim}** — {c[chr(99)+chr(111)+chr(109)+chr(117)+chr(110)+chr(97)+chr(115)]}")
    with ma:
        m=folium.Map(location=[-34.98,-71.08],zoom_start=10)
        for alim,c in sorted(info_alim.items()):
            col=COLORES_ALIM.get(alim,"gray")
            paso=max(1,len(c["pts"])//60)
            for i in range(0,len(c["pts"]),paso):
                folium.CircleMarker(c["pts"][i],radius=3,color=col,
                    fill=True,fill_opacity=0.35,weight=0).add_to(m)
            folium.Marker([c["lat"],c["lon"]],tooltip=f"Alimentador {alim}",
                popup=folium.Popup(
                    f"<b>Alim.{alim}</b><br>"
                    f"Comunas:{c[chr(99)+chr(111)+chr(109)+chr(117)+chr(110)+chr(97)+chr(115)]}<br>"
                    f"Fallas:{c[chr(110)+chr(95)+chr(102)+chr(97)+chr(108)+chr(108)+chr(97)+chr(115)]}",
                    max_width=180),
                icon=folium.Icon(color=col,icon="info-sign",prefix="glyphicon")).add_to(m)
        st_folium(m,width=700,height=500)

# ══════════════════════════════════════════════════════════
# PÁGINA 4 — AYUDA
# ══════════════════════════════════════════════════════════
elif pagina=="ℹ️ Ayuda":
    st.markdown("## ℹ️ Guía de uso")
    with st.expander("📝 ¿Cómo registrar una falla?",expanded=True):
        st.markdown("""
        1. Ve a **📝 Registrar Falla**
        2. Ingresa tu **nombre** y selecciona tu **cuadrilla**
        3. Elige **Guiado** (categoría + subcausa) o **Manual** (texto libre)
        4. Si tienes coordenadas X e Y, ingrésalas (acepta punto y coma)
        5. Presiona **🔍 Clasificar falla**
        6. Revisa el resultado y presiona **💾 Guardar registro**
        """)
    with st.expander("📊 Tipos de falla"):
        st.markdown("""
        | Tipo | Código | Descripción |
        |------|--------|-------------|
        | 🔴 Fuerza Mayor | FM | Accidentes, vandalismo, catástrofes |
        | 🟡 Externa | E | Fallas en generación o transmisión |
        | 🟢 Interna | I | Fallas propias de la red CEC |
        """)
    with st.expander("📍 Coordenadas y alimentadores"):
        st.markdown("""
        - Usa las coordenadas **X e Y** del Excel (UTM zona 19S)
        - Acepta tanto punto (315569.5) como coma (315569,5)
        - El modelo KNN identifica el alimentador con ~90% de precisión
        - 🟢 Alta (>80%) / 🟡 Media (50-80%) / 🔴 Baja (<50%)
        """)
    with st.expander("💾 ¿Dónde se guardan los datos?"):
        st.markdown("""
        Los registros se guardan en **GitHub** en el archivo `fallas_ingresadas.csv`.
        Puedes descargarlo como **CSV o Excel** desde la página **📋 Historial**.
        """)
