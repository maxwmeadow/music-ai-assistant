import { supabase } from '@/lib/supabase';
import { ProjectFile } from '@/lib/export';

export interface CloudProject {
    id: string;
    user_id: string;
    name: string;
    data: ProjectFile;
    created_at: string;
    updated_at: string;
}

export const CloudService = {
    async saveProject(name: string, projectData: ProjectFile, id?: string) {
        const user = (await supabase.auth.getUser()).data.user;
        if (!user) throw new Error('User not authenticated');

        const projectToSave = {
            name,
            data: projectData,
            user_id: user.id,
            updated_at: new Date().toISOString(),
        };

        if (id) {
            // Update existing
            const { data, error } = await supabase
                .from('projects')
                .update(projectToSave)
                .eq('id', id)
                .select()
                .single();

            if (error) throw error;
            return data as CloudProject;
        } else {
            // Create new
            const { data, error } = await supabase
                .from('projects')
                .insert([projectToSave])
                .select()
                .single();

            if (error) throw error;
            return data as CloudProject;
        }
    },

    async listProjects() {
        const { data, error } = await supabase
            .from('projects')
            .select('id, name, created_at, updated_at')
            .order('updated_at', { ascending: false });

        if (error) throw error;
        return data as Pick<CloudProject, 'id' | 'name' | 'created_at' | 'updated_at'>[];
    },

    async deleteProject(id: string) {
        const { error } = await supabase
            .from('projects')
            .delete()
            .eq('id', id);

        if (error) throw error;
    },

    async getProject(id: string) {
        const { data, error } = await supabase
            .from('projects')
            .select('*')
            .eq('id', id)
            .single();

        if (error) throw error;
        return data as CloudProject;
    }
};
