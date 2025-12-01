import { useState, useEffect } from 'react';
// Force rebuild
import { CloudService, CloudProject } from '@/services/cloudService';
import { ProjectFile } from '@/lib/export';
import { X, Save, Download, Trash2, Loader2, Cloud, Clock } from 'lucide-react';
import { toast } from 'sonner';
import { useAuth } from '@/contexts/AuthContext';

interface CloudProjectModalProps {
    isOpen: boolean;
    onClose: () => void;
    mode: 'save' | 'load';
    currentProjectData?: ProjectFile;
    onLoadProject?: (project: ProjectFile) => void;
}

export function CloudProjectModal({
    isOpen,
    onClose,
    mode,
    currentProjectData,
    onLoadProject
}: CloudProjectModalProps) {
    const { user } = useAuth();
    const [projects, setProjects] = useState<Pick<CloudProject, 'id' | 'name' | 'updated_at'>[]>([]);
    const [loading, setLoading] = useState(false);
    const [projectName, setProjectName] = useState('');
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        if (isOpen && user?.id) {
            loadProjects();
        }
    }, [isOpen, user?.id]);

    const loadProjects = async () => {
        setLoading(true);
        try {
            const list = await CloudService.listProjects();
            setProjects(list);
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!projectName.trim() || !currentProjectData) return;

        setSaving(true);
        try {
            await CloudService.saveProject(projectName, currentProjectData);
            toast.success('Project saved to cloud');
            onClose();
        } catch (error: any) {
            toast.error(error.message || 'Failed to save project');
        } finally {
            setSaving(false);
        }
    };

    const handleLoad = async (id: string) => {
        setLoading(true);
        try {
            const project = await CloudService.getProject(id);
            if (onLoadProject) {
                onLoadProject(project.data);
                onClose();
            }
        } catch (error) {
            console.error('Failed to load project:', error);
            toast.error('Failed to load project');
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!confirm('Are you sure you want to delete this project?')) return;

        try {
            await CloudService.deleteProject(id);
            toast.success('Project deleted');
            loadProjects(); // Refresh list
        } catch (error) {
            toast.error('Failed to delete project');
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="bg-[#252525] border border-gray-700 rounded-xl p-6 max-w-2xl w-full mx-4 shadow-2xl relative flex flex-col max-h-[80vh]">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
                >
                    <X size={20} />
                </button>

                <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                    <Cloud className="text-blue-400" />
                    {mode === 'save' ? 'Save to Cloud' : 'Cloud Projects'}
                </h2>

                {!user ? (
                    <div className="text-center py-12 text-gray-400">
                        Please log in to use cloud features.
                    </div>
                ) : (
                    <div className="flex-1 overflow-hidden flex flex-col">
                        {mode === 'save' && (
                            <form onSubmit={handleSave} className="mb-6 flex-none">
                                <label className="block text-xs font-medium text-gray-400 mb-1">Project Name</label>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={projectName}
                                        onChange={(e) => setProjectName(e.target.value)}
                                        className="flex-1 bg-[#1e1e1e] border border-gray-700 rounded-lg py-2 px-4 text-white text-sm focus:outline-none focus:border-blue-500 transition-colors"
                                        placeholder="My Awesome Song"
                                        required
                                        autoFocus
                                    />
                                    <button
                                        type="submit"
                                        disabled={saving}
                                        className="bg-blue-600 hover:bg-blue-500 text-white font-medium px-6 py-2 rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50"
                                    >
                                        {saving ? <Loader2 className="animate-spin" size={16} /> : <Save size={16} />}
                                        Save
                                    </button>
                                </div>
                            </form>
                        )}

                        <div className="flex-1 overflow-y-auto min-h-[300px] max-h-[400px] border border-gray-700 rounded-lg bg-[#1e1e1e] relative">
                            {loading ? (
                                <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500 gap-2">
                                    <Loader2 className="animate-spin w-8 h-8 text-blue-500" />
                                    <span>Loading projects...</span>
                                </div>
                            ) : projects.length === 0 ? (
                                <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500 gap-2">
                                    <Cloud size={32} className="opacity-20" />
                                    <span>No projects found</span>
                                </div>
                            ) : (
                                <div className="divide-y divide-gray-800">
                                    {projects.map((project) => (
                                        <div
                                            key={project.id}
                                            onClick={() => mode === 'load' && handleLoad(project.id)}
                                            className={`p-4 flex items-center justify-between group transition-colors ${mode === 'load' ? 'hover:bg-[#2a2a2a] cursor-pointer' : ''
                                                }`}
                                        >
                                            <div>
                                                <div className="font-medium text-white mb-1">{project.name}</div>
                                                <div className="text-xs text-gray-500 flex items-center gap-1">
                                                    <Clock size={12} />
                                                    {new Date(project.updated_at).toLocaleDateString()} {new Date(project.updated_at).toLocaleTimeString()}
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                {mode === 'save' && (
                                                    <button
                                                        onClick={() => setProjectName(project.name)}
                                                        className="p-2 text-gray-400 hover:text-blue-400 transition-colors"
                                                        title="Use this name"
                                                    >
                                                        <Download size={16} />
                                                    </button>
                                                )}
                                                <button
                                                    onClick={(e) => handleDelete(project.id, e)}
                                                    className="p-2 text-gray-400 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                                                    title="Delete project"
                                                >
                                                    <Trash2 size={16} />
                                                </button>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
