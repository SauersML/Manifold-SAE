import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import type { ManifoldPoint } from "@shared/schema";

export type ColorMode = "rgb" | "hue" | "modifier" | "monoword";

interface Props {
  points: ManifoldPoint[];
  colorMode: ColorMode;
  onPick: (id: number | null) => void;
  selectedId: number | null;
}

/** Three.js scene with InstancedMesh + GPU-side picking via a second
 *  picking scene that encodes instance index in RGB. */
export function ManifoldView({ points, colorMode, onPick, selectedId }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    renderer: THREE.WebGLRenderer;
    pickRT: THREE.WebGLRenderTarget;
    scene: THREE.Scene;
    pickScene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    mesh: THREE.InstancedMesh;
    pickMesh: THREE.InstancedMesh;
    highlight: THREE.Mesh;
    dispose: () => void;
  } | null>(null);

  const [hoverInfo, setHoverInfo] = useState<{
    x: number;
    y: number;
    p: ManifoldPoint;
  } | null>(null);

  // Init scene once
  useEffect(() => {
    const mount = mountRef.current!;
    const w = mount.clientWidth;
    const h = mount.clientHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h);
    renderer.setClearColor(0x05070b, 1);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const pickScene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 100);
    camera.position.set(2.5, 2.0, 2.5);
    camera.lookAt(0, 0, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(3, 4, 2);
    scene.add(dir);

    // Axes helper
    const axes = new THREE.AxesHelper(0.8);
    scene.add(axes);
    const grid = new THREE.GridHelper(4, 16, 0x222533, 0x161922);
    grid.position.y = -1.0;
    scene.add(grid);

    // InstancedMesh for points
    const N = Math.max(points.length, 1);
    const geom = new THREE.SphereGeometry(0.025, 12, 12);
    const mat = new THREE.MeshLambertMaterial({ vertexColors: false });
    const mesh = new THREE.InstancedMesh(geom, mat, N);
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(N * 3), 3);
    scene.add(mesh);

    // Picking mesh: same geometry, flat unlit shader, color = instance id encoded RGB
    const pickGeom = geom.clone();
    const pickMat = new THREE.MeshBasicMaterial({ vertexColors: false });
    const pickMesh = new THREE.InstancedMesh(pickGeom, pickMat, N);
    pickMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(N * 3), 3);
    pickScene.add(pickMesh);

    // Highlight ring
    const highlightGeom = new THREE.RingGeometry(0.05, 0.07, 32);
    const highlightMat = new THREE.MeshBasicMaterial({
      color: 0x22d3ee,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.0,
    });
    const highlight = new THREE.Mesh(highlightGeom, highlightMat);
    scene.add(highlight);

    const pickRT = new THREE.WebGLRenderTarget(w, h, { type: THREE.UnsignedByteType });

    // Camera controls (lightweight orbit, no external dep)
    let isDown = false;
    let isPan = false;
    let lastX = 0,
      lastY = 0;
    const target = new THREE.Vector3(0, 0, 0);
    let radius = camera.position.distanceTo(target);
    const spherical = new THREE.Spherical().setFromVector3(camera.position.clone().sub(target));

    const updateCam = () => {
      const v = new THREE.Vector3().setFromSpherical(spherical);
      camera.position.copy(target).add(v);
      camera.lookAt(target);
    };

    const dom = renderer.domElement;
    dom.addEventListener("mousedown", (e) => {
      isDown = true;
      isPan = e.shiftKey || e.button === 2;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    dom.addEventListener("contextmenu", (e) => e.preventDefault());
    window.addEventListener("mouseup", () => {
      isDown = false;
    });
    window.addEventListener("mousemove", (e) => {
      if (!isDown) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      if (isPan) {
        const panSpeed = radius * 0.0015;
        const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
        const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
        target.addScaledVector(right, -dx * panSpeed);
        target.addScaledVector(up, dy * panSpeed);
      } else {
        spherical.theta -= dx * 0.005;
        spherical.phi = Math.max(0.05, Math.min(Math.PI - 0.05, spherical.phi - dy * 0.005));
      }
      updateCam();
    });
    dom.addEventListener("wheel", (e) => {
      e.preventDefault();
      radius *= 1 + e.deltaY * 0.001;
      radius = Math.max(0.3, Math.min(20, radius));
      spherical.radius = radius;
      updateCam();
    });

    // Hover + click via GPU picking
    const pickAt = (clientX: number, clientY: number): number | null => {
      const rect = dom.getBoundingClientRect();
      const px = Math.floor((clientX - rect.left) * window.devicePixelRatio);
      const py = Math.floor((rect.height - (clientY - rect.top)) * window.devicePixelRatio);
      renderer.setRenderTarget(pickRT);
      renderer.render(pickScene, camera);
      const buf = new Uint8Array(4);
      renderer.readRenderTargetPixels(pickRT, px, py, 1, 1, buf);
      renderer.setRenderTarget(null);
      if (buf[3] === 0) return null;
      const id = buf[0] + (buf[1] << 8) + (buf[2] << 16) - 1;
      if (id < 0 || id >= points.length) return null;
      return id;
    };

    dom.addEventListener("click", (e) => {
      if (Math.abs(e.clientX - lastX) > 3 || Math.abs(e.clientY - lastY) > 3) return;
      const id = pickAt(e.clientX, e.clientY);
      onPick(id);
    });
    dom.addEventListener("mousemove", (e) => {
      if (isDown) return;
      const id = pickAt(e.clientX, e.clientY);
      if (id === null) {
        setHoverInfo(null);
      } else {
        const rect = dom.getBoundingClientRect();
        setHoverInfo({ x: e.clientX - rect.left, y: e.clientY - rect.top, p: points[id] });
      }
    });

    const onResize = () => {
      const ww = mount.clientWidth;
      const hh = mount.clientHeight;
      renderer.setSize(ww, hh);
      pickRT.setSize(ww * window.devicePixelRatio, hh * window.devicePixelRatio);
      camera.aspect = ww / hh;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);

    // Animation loop
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      renderer.render(scene, camera);
    };
    tick();

    stateRef.current = {
      renderer,
      pickRT,
      scene,
      pickScene,
      camera,
      mesh,
      pickMesh,
      highlight,
      dispose: () => {
        cancelAnimationFrame(raf);
        window.removeEventListener("resize", onResize);
        renderer.dispose();
        geom.dispose();
        pickGeom.dispose();
        highlightGeom.dispose();
        mat.dispose();
        pickMat.dispose();
        (highlightMat as THREE.Material).dispose();
        pickRT.dispose();
        mount.removeChild(renderer.domElement);
      },
    };
    return () => {
      stateRef.current?.dispose();
      stateRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-upload instance transforms whenever points change
  useEffect(() => {
    const st = stateRef.current;
    if (!st) return;
    const N = points.length;
    if (st.mesh.count !== N) {
      st.mesh.count = N;
      st.pickMesh.count = N;
    }
    const tmp = new THREE.Object3D();
    // Normalise XYZ to unit-ish box for nicer framing
    let lo = [Infinity, Infinity, Infinity];
    let hi = [-Infinity, -Infinity, -Infinity];
    for (const p of points) {
      for (let a = 0; a < 3; a++) {
        if (p.xyz[a] < lo[a]) lo[a] = p.xyz[a];
        if (p.xyz[a] > hi[a]) hi[a] = p.xyz[a];
      }
    }
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
    const scale = 2 / span;
    const cx = (hi[0] + lo[0]) / 2,
      cy = (hi[1] + lo[1]) / 2,
      cz = (hi[2] + lo[2]) / 2;

    for (let i = 0; i < N; i++) {
      const p = points[i];
      tmp.position.set((p.xyz[0] - cx) * scale, (p.xyz[1] - cy) * scale, (p.xyz[2] - cz) * scale);
      tmp.updateMatrix();
      st.mesh.setMatrixAt(i, tmp.matrix);
      st.pickMesh.setMatrixAt(i, tmp.matrix);
      // Encode (i+1) into RGB so 0 = background
      const id = i + 1;
      const pr = (id & 0xff) / 255;
      const pg = ((id >> 8) & 0xff) / 255;
      const pb = ((id >> 16) & 0xff) / 255;
      st.pickMesh.setColorAt(i, new THREE.Color(pr, pg, pb));
    }
    st.mesh.instanceMatrix.needsUpdate = true;
    st.pickMesh.instanceMatrix.needsUpdate = true;
    if (st.pickMesh.instanceColor) st.pickMesh.instanceColor.needsUpdate = true;
  }, [points]);

  // Re-color when colorMode changes
  useEffect(() => {
    const st = stateRef.current;
    if (!st) return;
    const c = new THREE.Color();
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      switch (colorMode) {
        case "rgb":
          c.setRGB(p.rgb[0], p.rgb[1], p.rgb[2]);
          break;
        case "hue":
          c.setHSL(p.hsv[0], 0.9, 0.55);
          break;
        case "modifier": {
          const t = Math.min(1, p.modifier_count / 3);
          c.setRGB(0.2 + 0.8 * t, 0.3, 1.0 - 0.7 * t);
          break;
        }
        case "monoword":
          c.setRGB(p.monoword ? 0.2 : 1.0, p.monoword ? 1.0 : 0.4, p.monoword ? 0.6 : 0.2);
          break;
      }
      st.mesh.setColorAt(i, c);
    }
    if (st.mesh.instanceColor) st.mesh.instanceColor.needsUpdate = true;
  }, [points, colorMode]);

  // Move highlight ring to selection
  useEffect(() => {
    const st = stateRef.current;
    if (!st) return;
    if (selectedId === null || !points[selectedId]) {
      (st.highlight.material as THREE.MeshBasicMaterial).opacity = 0;
      return;
    }
    const m = new THREE.Matrix4();
    st.mesh.getMatrixAt(selectedId, m);
    const pos = new THREE.Vector3().setFromMatrixPosition(m);
    st.highlight.position.copy(pos);
    st.highlight.lookAt(st.camera.position);
    (st.highlight.material as THREE.MeshBasicMaterial).opacity = 1;
  }, [selectedId, points]);

  return (
    <>
      <div ref={mountRef} style={{ position: "absolute", inset: 0 }} />
      {hoverInfo && (
        <div
          className="tooltip"
          style={{ left: hoverInfo.x + 12, top: hoverInfo.y + 12 }}
        >
          <div>
            <span
              style={{
                display: "inline-block",
                width: 10,
                height: 10,
                background: hoverInfo.p.hex,
                borderRadius: 2,
                marginRight: 6,
              }}
            />
            <b>{hoverInfo.p.name}</b>
          </div>
          <div style={{ color: "#8a90a3" }}>
            id={hoverInfo.p.id} mods={hoverInfo.p.modifier_count}
            {hoverInfo.p.monoword ? " mono" : ""}
          </div>
        </div>
      )}
    </>
  );
}
